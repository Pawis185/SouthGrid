"""Run installed official Task3 methods offline, without constructing a service.
No copied score formulas, network clients, attempts or jitter seeds.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
from typing import Optional
import numpy as np

TASK = 'task3_tool_storage'
ROBOT = 'g1_omnipicker'
METHODS = {'_auto_score_all_steps', '_observe_frame_recursive', '_get_site_pos',
           '_get_body_pos', '_get_body_velocity', '_compute_task3_movement_metrics',
           '_is_task3_no_movement', '_apply_task3_penalties', '_get_attempt_duration_s'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def installed_paths():
    spec = importlib.util.find_spec('scorer_service')
    root = Path(next(iter(spec.submodule_search_locations))).parent
    return root / 'scorer_service/engine.py', root / 'configs/tasks.yaml', root / 'configs/robots.yaml'

def fingerprint():
    engine, tasks, robots = installed_paths()
    root = engine.parent.parent
    paths = [engine, tasks, robots]
    for name in ('scorer_service/frame_oracle.py', 'orca_competition', 'tasks'):
        p = root / name
        paths.extend(sorted(p.rglob('*.py')) if p.is_dir() else [p])
    return {str(p.relative_to(root)): sha(p) for p in paths}

def config():
    import yaml
    from orca_competition.condition_parser import get_robot_profile
    _, tasks, robots = installed_paths()
    task = next(t for t in yaml.safe_load(tasks.read_text())['tasks'] if t['task_id'] == TASK)
    profile = get_robot_profile(ROBOT, str(robots))
    sites = {s['step_id']: s['success']['all'][0]['site_distance'][0] for s in task['steps'][1:]}
    return task, profile, sites

def official_class():
    from scorer_service.frame_oracle import FrameData, FrameOracle
    from orca_competition.condition_parser import get_robot_profile
    engine, _, _ = installed_paths()
    tree = ast.parse(engine.read_text())
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and
               any(isinstance(n, ast.FunctionDef) and n.name == '_auto_score_all_steps' for n in c.body))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    if {n.name for n in methods} != METHODS:
        raise RuntimeError('Installed engine has changed: offline adapter needs review')
    constants = [n for n in tree.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id.startswith('_TASK3_') for t in n.targets)]
    module = ast.fix_missing_locations(ast.Module(body=constants + [ast.ClassDef(
        name='OfficialOffline', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[]))
    ns = dict(np=np, Optional=Optional, FrameData=FrameData, FrameOracle=FrameOracle,
              get_robot_profile=get_robot_profile)
    exec(compile(module, str(engine) + '::unmodified_offline_methods', 'exec'), ns)
    return ns['OfficialOffline']

def frame_from_record(r):
    from scorer_service.frame_oracle import FrameData
    return FrameData(ts=r['host_wall_ns'] / 1e9,
        site_positions={n: {'xpos': np.asarray(v['xpos']), 'xmat': np.asarray(v['xmat']).reshape(3, 3)}
                        for n, v in r['site_positions'].items()},
        body_positions={n: np.asarray(v) for n, v in r['body_positions'].items()},
        body_velocities={n: np.asarray(v) for n, v in r['body_velocities'].items()},
        contacts=r['contacts'], joint_positions={n: np.asarray(v) for n, v in r['joint_positions'].items()},
        joint_velocities={n: np.asarray(v) for n, v in r['joint_velocities'].items()})

def validate_frames(frames):
    _, profile, sites = config()
    required_sites = set(sites.values()) | {'Group_Interactive_ToolBox_base_site', profile['ee_site'], profile['ee_site_right']}
    last = None
    for r in frames:
        if not np.isfinite([r['sim_time'],r['host_wall_ns'],r['host_mono_ns']]).all():
            raise ValueError('invalid timestamp')
        for key in ('joint_positions','joint_velocities'):
            if any(not np.isfinite(v).all() for v in r[key].values()):
                raise ValueError('invalid joint values')
        for n in required_sites:
            v = r['site_positions'][n]
            if np.shape(v['xpos']) != (3,) or np.size(v['xmat']) != 9 or not np.isfinite(v['xpos']).all() or not np.isfinite(v['xmat']).all():
                raise ValueError('invalid site ' + n)
        for key in ('body_positions', 'body_velocities'):
            v = r[key][profile['base_body']]
            if np.shape(v) != (3,) or not np.isfinite(v).all():
                raise ValueError('invalid base ' + key)
        if last is not None:
            if r['sample_id'] != last['sample_id'] + 1:
                raise ValueError('missing physics samples')
            if r['sim_time'] <= last['sim_time'] or r['host_mono_ns'] <= last['host_mono_ns'] or r['host_wall_ns'] <= last['host_wall_ns']:
                raise ValueError('non-increasing clock / reset during episode')
            # Wall clock jumps invalidate duration-based scoring; never silently repair timestamps.
            if abs((r['host_wall_ns'] - last['host_wall_ns']) - (r['host_mono_ns'] - last['host_mono_ns'])) > 50_000_000:
                raise ValueError('host wall clock discontinuity')
        last = r

def score(frames):
    if not frames:
        raise ValueError('no recorded physics samples')
    validate_frames(frames)
    from tasks.task_chain_builder import build_task_chain
    _, tasks, robots = installed_paths()
    ref = official_class()()
    ref._task_id, ref._robot_id = TASK, ROBOT
    ref._robots_config_path = str(robots)
    ref._frame_samples = [frame_from_record(r) for r in frames]
    ref._step_results, ref._step_extra = [], {}
    ref._chain = build_task_chain(TASK, config_path=str(tasks), robot_id=ROBOT, robots_config_path=str(robots))
    ref._auto_score_all_steps()
    ref._apply_task3_penalties()
    return dict(total=round(sum(r['score'] for r in ref._step_results), 2), maximum=40,
                steps=ref._step_results, through_sample_id=frames[-1]['sample_id'],
                through_mono_ns=frames[-1]['host_mono_ns'], frame_count=len(frames),
                scope='local official rules, before attempt-dependent final jitter; not leaderboard',
                time_basis='recorded host wall clock (same basis as official FrameData.ts)',
                official_attempts_sent=0)
