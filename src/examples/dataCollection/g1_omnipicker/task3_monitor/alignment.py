"""Metric world-space guidance; never changes the model or camera pixels."""
import json
from pathlib import Path
import numpy as np
BOX = 'Group_Interactive_ToolBox_base_site'
NAMES = {'pick_wrench':'扳手', 'pick_screwdriver':'螺丝刀', 'pick_electrician_knife':'电工刀01',
         'pick_flashlight':'手电筒', 'pick_voltage_pen':'电工刀02'}

def transform(position, rotation, local):
    return np.asarray(position) + np.asarray(rotation).reshape(3,3) @ np.asarray(local)

def markers(frame):
    data = json.loads(Path(__file__).with_name('alignment.json').read_text())
    sites = frame['site_positions']; b = sites[BOX]; result=[]
    for key, label in NAMES.items():
        entry=data['tools'][key]; region=entry['region_evidence']; s=sites[region['site']]
        grasp=entry['grasp_ee_in_tool_site']
        p=transform(s['xpos'], s['xmat'], grasp['position'])
        rotation=np.asarray(s['xmat']).reshape(3,3) @ np.asarray(grasp['rotation'])
        release=transform(b['xpos'],b['xmat'],entry['release_tool_in_box_site'])
        result.append(dict(key=key,label=label,current=s['xpos'],grasp=p.tolist(),
            grasp_axes=rotation.tolist(),approach=(p+np.array([0,0,.10])).tolist(),
            peak=b['xpos'],release=release.tolist(),distance_mm=float(np.linalg.norm(np.asarray(s['xpos'])-b['xpos'])*1000),
            theoretical_radius_m=region['theoretical_region']['approximate_peak_radius_m'],
            observed_region=region.get('observed_scoring_peak_box_relative_xyz'),
            recommended=region.get('recommended_control_ranges'),
            provenance=entry['source_sha256'],
            guidance_status='observed grasp/release transferred to current pose; approach +10cm is unvalidated clearance guidance'))
    return dict(box=b['xpos'],tools=result,reach_shell_m=[.649375,.65175],
                reach_note='3D distance shell, not horizontal radius; rounded 5/5 theoretical interval',
                base=frame['body_positions']['g1_omnipicker_body_link1'],
                ee=sites['g1_omnipicker_ee_center_site_r']['xpos'],
                note=data['note'])
