"""Independent host monitor + offline rescoring CLI. No ORCA connections."""
import argparse
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
from .journal import atomic, dumps, read_records
from .scoring import score, fingerprint, config
from .alignment import markers

def review(path, final=False):
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['metadata']['scoring_fingerprint'] != fingerprint():
        raise ValueError('scoring version differs from collection manifest')
    records,torn=read_records(path/'raw.jsonl')
    for i,r in enumerate(records):
        if r['record_id'] != i: raise ValueError('journal record gap')
        if r['kind']=='capture_error': raise ValueError('capture error: '+r['reason'])
    if final:
        if torn or not manifest.get('scoring_complete'): raise ValueError('incomplete/interrupted journal; final score unavailable')
        digest=json.loads((path/'raw_digest.json').read_text())
        if hashlib.sha256((path/'raw.jsonl').read_bytes()).hexdigest()!=digest['sha256']:
            raise ValueError('raw journal checksum mismatch')
        if not records or records[-1]['kind']!='episode_end' or records[-1]['record_id']!=manifest['last_record_id']:
            raise ValueError('missing episode end')
    frames=[r for r in records if r['kind']=='physics']
    ids={r['sample_id'] for r in frames}
    captures={r['capture_id']:r for r in records if r['kind']=='camera_capture'}
    rows=[r for r in records if r['kind']=='lerobot_frame']
    for r in captures.values():
        if r['sample_id'] not in ids: raise ValueError('camera capture has no associated physics sample')
    for i,r in enumerate(rows):
        if r['frame_index']!=i: raise ValueError('LeRobot frame sequence gap')
        for key in ('observation_capture','action_target_capture'):
            c=r[key]
            if c['capture_id'] not in captures or c['sample_id'] not in ids:
                raise ValueError('broken frame/action association')
    if final and 'scene_mjb_sha256' in manifest['metadata']:
        if hashlib.sha256((path/'scene.mjb').read_bytes()).hexdigest()!=manifest['metadata']['scene_mjb_sha256']:
            raise ValueError('model checksum mismatch')
    with contextlib.redirect_stdout(io.StringIO()):
        result=score(frames)
    # Auxiliary observations are separate from official conditions/results.
    import numpy as np
    _, _, tools = config()
    names = manifest['metadata']['model_names']['geoms']
    last = frames[-1]
    tail = [f for f in frames if last['sim_time'] - f['sim_time'] <= 1.0]
    observations = {}
    for key, site in tools.items():
        prefix = site.split('_task_')[0]
        robot_contact = box_contact = False
        for c in last['contacts']:
            pair = [names[c[k]] for k in ('geom1','geom2')]
            if any(n.startswith(prefix) for n in pair):
                robot_contact |= any(n.startswith('g1_omnipicker') for n in pair)
                box_contact |= any(n.startswith('Group_Interactive_ToolBox') for n in pair)
        positions = np.array([f['site_positions'][site]['xpos'] for f in tail])
        enough = len(tail)>1 and last['sim_time']-tail[0]['sim_time'] >= .95
        observations[key] = dict(robot_contact=robot_contact, toolbox_contact=box_contact,
            release_status='no robot contact observed' if not robot_contact else 'robot contact observed (grasp not proven)',
            stable_last_sim_second=bool(enough and np.max(np.linalg.norm(positions-positions[-1],axis=1))<=.002),
            stable_definition='auxiliary: <=2mm over >=0.95 simulation seconds; not official predicate',
            distance_to_box_m=float(np.linalg.norm(positions[-1]-np.array(last['site_positions']['Group_Interactive_ToolBox_base_site']['xpos']))))
    result['observations'] = observations
    result['source'] = manifest['source']
    result['association_check'] = dict(camera_captures=len(captures), lerobot_frames=len(rows),
        duplicate_camera_captures=sum(bool(c.get('duplicate_camera_frames')) for c in captures.values()),
        exposure_timestamp_available=False)
    result.update(status='ok' ,kind='episode_review' if final else 'trajectory_prefix',
                  episode_uuid=manifest['episode_uuid'],computed_mono_ns=time.monotonic_ns(),
                  tail_incomplete=torn,scoring_fingerprint=manifest['metadata']['scoring_fingerprint'])
    return result

class State:
    def __init__(self,root):
        self.root=root;self.result={'status':'unavailable','reason':'等待录制开始'}
    def loop(self):
        done={}
        while True:
            try:
                current=json.loads((self.root/'current.json').read_text())['episode_uuid']
                path=self.root/'episodes'/current
                manifest=json.loads((path/'manifest.json').read_text())
                final=manifest['state']!='open'
                if final and current in done:
                    self.result=done[current]
                else:
                    self.result=review(path,final=final)
                    atomic(path/('review.json' if final else 'score_latest.json'),self.result)
                    from .preview import svg
                    latest=json.loads((path/'latest.json').read_text())
                    preview=path/'previews';preview.mkdir(exist_ok=True)
                    (preview/('sample_%09d.svg' % latest['sample_id'])).write_text(svg(latest,manifest['source']))
                    with (path/'score_snapshots.jsonl').open('a') as f:f.write(dumps(self.result)+'\n')
                    if final: done[current]=self.result
                # Also finalize preceding episodes when operator immediately starts another.
                for old in (self.root/'episodes').iterdir():
                    if old.name==current or old.name in done:continue
                    m=json.loads((old/'manifest.json').read_text())
                    if m['state']=='open':continue
                    try:r=review(old,final=True)
                    except Exception as e:r=dict(status='unavailable',reason=str(e),total=None)
                    atomic(old/'review.json',r);done[old.name]=r
            except Exception as e:
                self.result={'status':'unavailable','reason':str(e),'total':None}
            time.sleep(2)
    def payload(self):
        result={'score':dict(self.result)}
        try:
            current=json.loads((self.root/'current.json').read_text())['episode_uuid']
            p=self.root/'episodes'/current
            manifest=json.loads((p/'manifest.json').read_text())
            f=json.loads((p/'latest.json').read_text())
            result.update(episode=current,manifest=manifest,scene=markers(f),
                          scene_age_s=(time.monotonic_ns()-f['host_mono_ns'])/1e9,
                          sim_time=f['sim_time'],contacts=f['contacts'],
                          joints={k:v for k,v in f['joint_positions'].items() if 'gripper' in k},
                          state=('待分类：录制已结束，最终日志校验待提交' if manifest.get('recording_ended') else '录制中') if manifest['state']=='open' else manifest['state'])
            s=result['score']
            if s.get('episode_uuid')!=current:
                s.update(status='unavailable',total=None,reason='新 episode 尚未评分')
            elif s.get('kind')=='trajectory_prefix':
                age=(time.monotonic_ns()-s['through_mono_ns'])/1e9
                s['age_s']=age
                if age>5 and not manifest.get('recording_ended'):s.update(status='stale',total=None,reason='评分数据超过5秒未更新；历史快照保存在日志')
        except Exception as e:result['scene_error']=str(e)
        return result

def main():
    import os
    os.nice(10)  # Scoring yields CPU priority to teleoperation.
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--review',type=Path)
    args=parser.parse_args()
    if args.review:
        try:r=review(args.review,final=True)
        except Exception as e:r=dict(status='unavailable',total=None,reason=str(e))
        atomic(args.review/'review.json',r)
        print(json.dumps(r,ensure_ascii=False,indent=2));return
    state=State(args.root)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/api': data=dumps(state.payload()).encode();mime='application/json'
            elif self.path=='/':data=Path(__file__).with_name('monitor.html').read_bytes();mime='text/html; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime)
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    threading.Thread(target=state.loop,daemon=True).start()
    server.serve_forever()
if __name__=='__main__':main()
