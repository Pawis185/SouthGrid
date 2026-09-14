"""All generated trajectories here are synthetic tests, never ORCA evidence."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import numpy as np
import mujoco

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/examples/dataCollection/g1_omnipicker'))
from task3_monitor.journal import Journal, atomic, read_records
from task3_monitor.scoring import config, fingerprint, score
from task3_monitor.service import review
from task3_monitor.alignment import transform, markers
from task3_monitor.capture import Capture


def fixture(i):
    _,p,tools=config()
    sites={n:{'xpos':[0.,0.,0.], 'xmat':np.eye(3).tolist()}
           for n in [*tools.values(),'Group_Interactive_ToolBox_base_site',p['ee_site'],p['ee_site_right']]}
    sites[p['ee_site_right']]['xpos']=[i*.05,0.,.1]
    return dict(sample_id=i,sim_time=i*.5+1,host_wall_ns=1_000_000_000_000+i*1_000_000_000,
                host_mono_ns=1_000_000_000_000+i*1_000_000_000,
                site_positions=sites,body_positions={p['base_body']:[.65,0.,0.]},
                body_velocities={p['base_body']:[0.,0.,0.]},joint_positions={},joint_velocities={},contacts=[])

class Tests(unittest.TestCase):
    def test_transform_and_marker_frame(self):
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(transform([1,2,3],rot,[1,0,0]),[1,3,3])
        f=fixture(1); scene=markers(f)
        self.assertEqual(len(scene['tools']),5)
        self.assertEqual(scene['tools'][-1]['label'],'电工刀02')
        self.assertEqual(scene['tools'][-1]['peak'],[0.,0.,0.])

    def test_missing_inputs_clock_gap_and_record_gap_rejected(self):
        frames=[fixture(i) for i in range(3)]
        del frames[1]['site_positions'][config()[2]['pick_voltage_pen']]
        with self.assertRaises(KeyError):score(frames)
        frames=[fixture(i) for i in range(3)];frames[1]['sample_id']=10
        with self.assertRaisesRegex(ValueError,'missing physics'):score(frames)
        frames=[fixture(i) for i in range(3)];frames[1]['host_wall_ns']+=100_000_000
        with self.assertRaisesRegex(ValueError,'discontinuity'):score(frames)

    def test_journal_resume_torn_tail_and_rescore_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta={'scoring_fingerprint':fingerprint(),'model_names':{'geoms':[]},'test_data':True}
            journal=Journal(tmp,meta)
            frames=[fixture(i) for i in range(20)]
            for f in frames:journal.emit('physics',**f)
            journal.close('saved',lerobot_episode_index=0,lerobot_category='good')
            with contextlib.redirect_stdout(io.StringIO()):
                prefix=review(journal.path,final=False);final=review(journal.path,final=True)
            self.assertEqual(prefix['total'],final['total'])
            self.assertEqual(prefix['steps'],final['steps'])
            self.assertEqual(final['total'],40.)
            next_episode=Journal(tmp,meta)
            self.assertNotEqual(journal.id,next_episode.id)
            next_episode.emit('test_marker');next_episode.close('interrupted')
            with (next_episode.path/'raw.jsonl').open('ab') as f:f.write(b'{"torn":')
            records,torn=read_records(next_episode.path/'raw.jsonl')
            self.assertTrue(torn);self.assertEqual(records[0]['kind'],'test_marker')
            with self.assertRaisesRegex(ValueError,'incomplete'):review(next_episode.path,True)
            with (journal.path/'raw.jsonl').open('ab') as f:f.write(b'{}\n')
            with self.assertRaises((KeyError,ValueError)):review(journal.path,True)

    def test_actual_substeps_identical_and_contact_transient_captured(self):
        m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.002"/><worldbody>
          <geom type="plane" size="1 1 .1"/><body name="ball" pos="0 0 .11">
          <freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
        a,b=mujoco.MjData(m),mujoco.MjData(m)
        seen=[]
        for _ in range(100):
            mujoco.mj_step(m,a,nstep=5)
            for k in range(5):
                mujoco.mj_step(m,b,nstep=1)
                seen.append((b.time,b.ncon))
        np.testing.assert_array_equal(a.qpos,b.qpos)
        np.testing.assert_array_equal(a.qvel,b.qvel)
        self.assertEqual(len(seen),500)
        self.assertTrue(any(n>0 for _,n in seen));self.assertTrue(any(n==0 for _,n in seen))

    def test_capture_reads_physics_model_and_serializes_full_inputs(self):
        task, profile, tools = config()
        sites = [*tools.values(), 'Group_Interactive_ToolBox_base_site', profile['ee_site'], profile['ee_site_right']]
        xml = '<mujoco><option timestep="0.002"/><worldbody>'
        xml += ''.join('<site name="'+n+'" pos="0 0 0"/>' for n in sites)
        xml += '<body name="'+profile['base_body']+'" pos=".65 0 .5"><freejoint/><geom type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>'
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        with tempfile.TemporaryDirectory() as tmp:
            cap=Capture.__new__(Capture)
            cap.root=Path(tmp);cap.storage=SimpleNamespace();cap.args={'task_config':'missing-synthetic.yaml'}
            cap.version=fingerprint();cap.task=task;cap.profile=profile;cap.tools=tools
            cap.mj=mujoco;cap.journal=None
            cap.env=SimpleNamespace(gym=SimpleNamespace(_mjModel=m,_mjData=d))
            cap.manager=SimpleNamespace(task_status_controller=SimpleNamespace(current_status=SimpleNamespace(name='RUNNING')),
                device=SimpleNamespace(pico_joystick=SimpleNamespace(clients=[],current_key_state={})))
            cap.original_step=lambda n:mujoco.mj_step(m,d,nstep=n)
            cap.begin()
            cap.journal.manifest['source']='synthetic_test'
            cap.journal.manifest['metadata']['test_data']=True
            atomic(cap.journal.path/'manifest.json',cap.journal.manifest)
            for _ in range(40):cap.step(5)
            cap.journal.close('saved')
            records,torn=read_records(cap.journal.path/'raw.jsonl')
            frames=[r for r in records if r['kind']=='physics']
            self.assertFalse(torn);self.assertEqual(len(frames),200)
            self.assertEqual(len([r for r in records if r['kind']=='control_applied']),40)
            self.assertEqual(frames[-1]['sample_id'],199)
            self.assertAlmostEqual(frames[-1]['sim_time'],.4)
            self.assertTrue((cap.journal.path/'scene.mjb').is_file())
            with contextlib.redirect_stdout(io.StringIO()):r=review(cap.journal.path,True)
            self.assertEqual(r['status'],'ok')

    def test_frame_association_uses_previous_image_and_current_target(self):
        class Sink:
            def __init__(self):self.records=[]
            def emit(self,kind,**kw):self.records.append(dict(kind=kind,**kw))
        cap=Capture.__new__(Capture);cap.active=True;cap.journal=Sink();cap.sample_id=10;cap.control_id=2
        cap.prev_capture=None;cap.last_camera_indices={}
        cap.storage=SimpleNamespace(buffered_frame_count=0,_lr_prev=None)
        def collect(*args,**kwargs):
            cap.storage.buffered_frame_count+=1
            cap.storage._lr_prev=(None,None,{'cam_head':99})
        cap.original_collection=collect
        env=SimpleNamespace(data=SimpleNamespace(time=.1))
        cap.collection({},env);cap.sample_id=20;cap.control_id=4;cap.collection({},env)
        row=next(r for r in cap.journal.records if r['kind']=='lerobot_frame')
        self.assertEqual(row['frame_index'],0)
        self.assertEqual(row['observation_capture']['sample_id'],10)
        self.assertEqual(row['action_target_capture']['sample_id'],20)
        self.assertEqual(cap.journal.records[-2]['duplicate_camera_frames'],['cam_head'])

if __name__=='__main__':unittest.main()
