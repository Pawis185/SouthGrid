"""Read-only snapshots of the collecting environment's actual MuJoCo state."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .journal import Journal, atomic
from .scoring import config, fingerprint

class Capture:
    def __init__(self, root, args, env, manager, storage):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        import fcntl
        self.lock = (self.root/'collection.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for old in (self.root/'episodes').glob('*/manifest.json'):
            manifest = json.loads(old.read_text())
            if manifest['state'] == 'open':
                manifest.update(state='interrupted_before_resume', scoring_complete=False,
                                recovery_note='original raw.jsonl retained, including any torn tail')
                atomic(old, manifest)
        self.env, self.manager, self.storage = env, manager, storage
        self.journal = None
        self.sample_id = self.control_id = -1
        self.active = False
        self.last_status = None
        self.args = vars(args).copy()
        self.version = fingerprint()
        expected = json.loads(Path(__file__).with_name('scoring_version.json').read_text())
        if self.version != expected:
            raise RuntimeError('Official scoring version changed; adapter and alignment need review')
        self.task, self.profile, self.tools = config()
        import mujoco
        self.mj = mujoco
        self.original_step = env.gym.mj_step
        # Known local implementation consists of exactly one mj_step call.
        source = inspect.getsource(self.original_step)
        import ast, textwrap
        fn = ast.parse(textwrap.dedent(source)).body[0]
        executable = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        if len(executable) != 1 or ast.unparse(executable[0]) != 'mujoco.mj_step(self._mjModel, self._mjData, nstep)':
            raise RuntimeError('Unknown physics stepping implementation; review adapter before enabling')
        self.original_collection = storage.collection_data
        self.last_camera_indices = {}
        self.prev_capture = None
        self.previous_sample_mono_ns = None
        self.monitor = subprocess.Popen([sys.executable, '-m', 'task3_monitor.service',
            '--root', str(self.root), '--port', str(args.task3_monitor_port)],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=(self.root/'service.log').open('a'), stderr=subprocess.STDOUT)
        time.sleep(.3)
        if self.monitor.poll() is not None:
            raise RuntimeError('Task3 monitor could not start; inspect scene_sidecar/service.log')
        env.gym.mj_step = self.step
        storage.collection_data = self.collection

    def begin(self, tool_layout=None):
        if self.journal: self.finish('abandoned')
        self.sample_id = self.control_id = -1
        self.active = False
        self.last_status = None
        self.prev_capture = None
        self.last_camera_indices = {}
        self.previous_sample_mono_ns = None
        m = self.env.gym._mjModel
        def names(kind, n):
            return [self.mj.mj_id2name(m, kind, i) or f'unnamed_{i}' for i in range(n)]
        self.site_names = names(self.mj.mjtObj.mjOBJ_SITE, m.nsite)
        self.body_names = names(self.mj.mjtObj.mjOBJ_BODY, m.nbody)
        self.joint_names = names(self.mj.mjtObj.mjOBJ_JOINT, m.njnt)
        self.geom_names = names(self.mj.mjtObj.mjOBJ_GEOM, m.ngeom)
        # All named sites, robot and object bodies: no remote queries in the control loop.
        prefixes = tuple(s.split('_task_')[0] for s in self.tools.values()) + ('g1_omnipicker', 'Group_Interactive_ToolBox')
        self.body_ids = [i for i,n in enumerate(self.body_names) if n.startswith(prefixes)]
        missing = (set(self.tools.values()) | {self.profile['ee_site'], self.profile['ee_site_right'], 'Group_Interactive_ToolBox_base_site'}) - set(self.site_names)
        if missing: raise RuntimeError('Scene does not contain required Task3 sites: ' + repr(missing))
        meta = dict(args=self.args, scoring_fingerprint=self.version, task=self.task, robot_profile=self.profile,
                    tools=self.tools, units='metres, seconds, radians; world right-handed; matrices row-major; quaternions wxyz',
                    physics_timestep=float(m.opt.timestep), sampling='each local physics substep; unmodified controls and total substeps',
                    timestamp_basis='ts for scoring uses host wall clock, matching official engine; sim_time is independently recorded',
                    model_names=dict(sites=self.site_names, bodies=self.body_names, joints=self.joint_names, geoms=self.geom_names),
                    model_dimensions=dict(nq=m.nq,nv=m.nv,nu=m.nu),
                    camera_map=getattr(self.storage, '_lr_camera_map', {}),
                    collector_hashes={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.iterdir() if p.suffix in {'.py','.json','.html'}})
        entry = Path(__file__).parent.parent/'g1_omnipicker_collection_tele_lerobot.py'
        meta['entrypoint_sha256'] = hashlib.sha256(entry.read_bytes()).hexdigest()
        config_path = Path(self.args['task_config']).resolve()
        if config_path.is_file(): meta['task_config_text'] = config_path.read_text()
        self.journal = Journal(self.root, meta)
        # Persist the actual loaded model, including site/body configuration, without changing it.
        self.mj.mj_saveModel(m, str(self.journal.path/'scene.mjb'), None)
        meta['scene_mjb_sha256'] = hashlib.sha256((self.journal.path/'scene.mjb').read_bytes()).hexdigest()
        if tool_layout:
            meta['tool_layout'] = tool_layout
        self.journal.manifest['metadata'] = meta
        atomic(self.journal.path/'manifest.json', self.journal.manifest)
        reset_fields = dict(sim_time=float(self.env.gym._mjData.time))
        if tool_layout:
            reset_fields['tool_layout'] = tool_layout
        self.journal.emit('reset_completed', **reset_fields)
        self.snapshot(recording=False)
        atomic(self.root/'current.json', {'episode_uuid': self.journal.id})

    def step(self, nstep):
        if self.journal is None:
            return self.original_step(nstep)
        status = self.manager.task_status_controller.current_status.name
        if status != self.last_status:
            self.journal.emit('task_status', previous=self.last_status, current=status)
            self.last_status = status
        running = status == 'RUNNING'
        if running and not self.active:
            self.active = True
            self.journal.emit('recording_start', sim_time=float(self.env.gym._mjData.time))
        self.control_id += 1
        if not self.active:
            result = self.original_step(nstep)
            if time.monotonic() - getattr(self, 'last_idle_preview', 0.) > .2:
                self.last_idle_preview = time.monotonic()
                try: self.snapshot(recording=False)
                except Exception as exc: self.journal.emit('idle_preview_error', reason=str(exc))
            return result
        import copy
        self.journal.emit('control_applied', control_id=self.control_id,
            ctrl=self.env.gym._mjData.ctrl.copy(),
            pico_raw=copy.deepcopy(self.manager.device.pico_joystick.current_key_state),
            sim_time=float(self.env.gym._mjData.time), requested_substeps=int(nstep))
        for _ in range(nstep):
            self.original_step(1)
            try:
                self.snapshot()
            except Exception as exc:
                self.journal.dropped += 1
                self.journal.emit('capture_error', reason=str(exc))

    def snapshot(self, recording=True):
        m, d = self.env.gym._mjModel, self.env.gym._mjData
        if recording: self.sample_id += 1
        acquired_wall_ns, acquired_mono_ns = time.time_ns(), time.monotonic_ns()
        previous = getattr(self, 'previous_sample_mono_ns', None)
        if previous and acquired_mono_ns - previous > 250_000_000:
            self.journal.emit('sampling_pause_or_delay', duration_ns=acquired_mono_ns-previous,
                              reason='host sampling gap; no inferred external pause command')
        self.previous_sample_mono_ns = acquired_mono_ns
        sites = {n:dict(xpos=d.site_xpos[i].copy(), xmat=d.site_xmat[i].copy()) for i,n in enumerate(self.site_names)}
        joints, velocities = {}, {}
        for i,n in enumerate(self.joint_names):
            qend = m.jnt_qposadr[i+1] if i+1 < m.njnt else m.nq
            vend = m.jnt_dofadr[i+1] if i+1 < m.njnt else m.nv
            joints[n] = d.qpos[m.jnt_qposadr[i]:qend].copy()
            velocities[n] = d.qvel[m.jnt_dofadr[i]:vend].copy()
        contacts=[]
        for c in d.contact:
            contacts.append(dict(geom1=int(c.geom1), geom2=int(c.geom2), dist=float(c.dist),
                                 pos=c.pos.copy(), frame=c.frame.copy(), dim=int(c.dim)))
        self.journal.emit('physics' if recording else 'idle_preview', sample_id=self.sample_id, control_id=self.control_id,
            host_wall_ns=acquired_wall_ns, host_mono_ns=acquired_mono_ns,
            sim_time=float(d.time), ctrl=d.ctrl.copy(), qpos=d.qpos.copy(), qvel=d.qvel.copy(),
            act=d.act.copy(), mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy(),
            site_positions=sites, body_positions={self.body_names[i]:d.xpos[i].copy() for i in self.body_ids},
            body_quaternions={self.body_names[i]:d.xquat[i].copy() for i in self.body_ids},
            body_velocities={self.body_names[i]:d.cvel[i,3:].copy() for i in self.body_ids},
            body_angular_velocities={self.body_names[i]:d.cvel[i,:3].copy() for i in self.body_ids},
            joint_positions=joints, joint_velocities=velocities, contacts=contacts,
            pico_connected=bool(self.manager.device.pico_joystick.clients) if hasattr(self.manager,'device') else None)

    def collection(self, obs, env, **kwargs):
        before = self.storage.buffered_frame_count
        capture_start_mono_ns = time.monotonic_ns()
        try:
            self.original_collection(obs, env, **kwargs)
        except Exception as exc:
            if self.journal: self.journal.emit('camera_or_storage_error', reason=str(exc))
            raise
        after = self.storage.buffered_frame_count
        if self.journal is None or after == before: return
        if not self.active:
            self.active = True
            self.journal.emit('recording_start', sim_time=float(env.data.time))
            self.snapshot()
        prev = self.storage._lr_prev
        indices = dict(prev[2]) if prev is not None else {}
        duplicates = [k for k,v in indices.items() if self.last_camera_indices.get(k) == v]
        gaps = {k:v-self.last_camera_indices[k]-1 for k,v in indices.items()
                if k in self.last_camera_indices and v > self.last_camera_indices[k]+1}
        current = dict(capture_id=before, sample_id=self.sample_id, control_id=self.control_id,
                       sim_time=float(env.data.time), camera_frame_indices=indices,
                       capture_return_mono_ns=time.monotonic_ns(), capture_start_mono_ns=capture_start_mono_ns)
        self.journal.emit('camera_capture', **current, duplicate_camera_frames=duplicates,
                          skipped_source_frames=gaps, exposure_timestamp=None,
                          note='receiver frame index only; exposure time unavailable from existing wrapper')
        if self.prev_capture is not None and prev is not None:
            self.journal.emit('lerobot_frame', frame_index=before-1, observation_capture=self.prev_capture,
                              action_target_capture=current, note='LeRobot action is next captured state; actual actuator ctrl is in physics records')
        self.prev_capture = current
        self.last_camera_indices = indices

    def recording_end(self):
        if self.journal:
            self.journal.emit('recording_end', sim_time=float(self.env.data.time))
            self.journal.manifest['recording_ended'] = True
            atomic(self.journal.path/'manifest.json', self.journal.manifest)
        self.active = False

    def finish(self, outcome, **fields):
        if self.journal:
            self.journal.close(outcome, **fields)
            subprocess.Popen([sys.executable, '-m', 'task3_monitor.service', '--review', str(self.journal.path)],
                cwd=str(Path(__file__).resolve().parent.parent),
                stdout=(self.journal.path/'review_worker.log').open('a'), stderr=subprocess.STDOUT)
            self.journal = None
        self.active = False

    def close(self):
        self.finish('interrupted')
        self.env.gym.mj_step = self.original_step
        self.storage.collection_data = self.original_collection
        # Stop only our own read-only service, never any ORCA/Pico process.
        self.monitor.terminate()
        try: self.monitor.wait(timeout=3)
        except subprocess.TimeoutExpired: pass
        self.lock.close()
