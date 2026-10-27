"""Hardware-free full WORK worker benchmark with real camera/recording processes.

Only HDAPI/UDP source and frontend consumer are substituted. This is NOT a
physical UI acceptance test. No socket, Touch, launcher or browser is started.
All images and discarded episodes stay in /tmp. Optional baseline is a snapshot.
"""
import argparse
from collections import deque
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import sys
import tempfile
import time
from unittest.mock import patch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds-per-phase',type=float,default=4.)
    ap.add_argument('--idle-seconds',type=float,help='optional shorter stationary lead-in; active phase duration is unchanged')
    ap.add_argument('--diagnostics',choices=('off','production','profile'),default='off')
    ap.add_argument('--record',action='store_true')
    ap.add_argument('--rotation-mode',choices=('rotation','rotation_z','6dof'),default='rotation')
    ap.add_argument('--sync',action='store_true',help='bounded raw synchronization capture with finite step/hold inputs')
    ap.add_argument('--baseline-root',type=Path)
    ap.add_argument('--camera-size',type=int,nargs=2,default=(960,1280))
    args=ap.parse_args()
    idle_seconds=args.seconds_per_phase if args.idle_seconds is None else args.idle_seconds
    if not math.isfinite(idle_seconds) or idle_seconds<=0:ap.error('idle-seconds must be positive')
    if not math.isfinite(args.seconds_per_phase) or args.seconds_per_phase<=0:ap.error('seconds-per-phase must be positive')
    if args.sync:
        os.environ['GEOMAGIC_SYNC_CAPTURE']=str(args.output.with_suffix('.capture').resolve())
        os.environ['GEOMAGIC_SYNC_SECONDS']=str(min(600.,idle_seconds+2*args.seconds_per_phase+65.))
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ['MIMIC_GEOMAGIC_DIAGNOSTICS'] = '0' if args.diagnostics == 'off' else '1'
    os.environ['MIMIC_GEOMAGIC_DIAGNOSTICS_MODE']='detailed' if args.diagnostics=='profile' else args.diagnostics
    os.environ['MIMIC_GEOMAGIC_PROFILE']='1' if args.diagnostics=='profile' else '0'
    os.environ['MIMIC_GEOMAGIC_PROFILE_REPORT']=str(args.output.with_suffix('.profile.json').resolve())
    os.environ['MIMIC_GEOMAGIC_MEASURE_ONLY']='0' if args.record else '1'
    from geomagic.teleop.robot_helpers import MIMIC_ROOT;sys.path.insert(0,str(MIMIC_ROOT))
    if args.baseline_root:
        for name in ('geomagic.teleop.pose_mapper','geomagic.teleop.diagnostics','geomagic.leader.teleop_backend'):
            p=args.baseline_root/(name.replace('.','/')+'.py')
            spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    import numpy as np
    from geomagic.leader.teleop_backend import GeomagicTeleopBackend
    from geomagic.bridge.udp_state_client import GeomagicUDPStateClient
    from geomagic.bridge.protocol import GeomagicStatePacket,encode_packet
    from runtime import ui_worker,recording
    from runtime.live_runner import MimicLiveRunner
    from geomagic.teleop.runtime_profile import get_profiler
    if args.baseline_root:
        name='runtime.ui_worker_baseline';p=args.baseline_root/'mimic/runtime/ui_worker.py'
        spec=importlib.util.spec_from_file_location(name,p);ui_worker=importlib.util.module_from_spec(spec);sys.modules[name]=ui_worker;spec.loader.exec_module(ui_worker)

    state={'start':None,'runner':None,'seq':0,'next_input':None,'command_phase':None,'done':False}
    rows=[];payloads=[];profile=get_profiler()
    if args.baseline_root and profile:
        profile.instrument(ui_worker, {'_timing':'ui.percentile_stats'})
        import geomagic.vendor.geomagic_core.rotations as vendor_rotations
        import geomagic.vendor.geomagic_core.controller as vendor_controller
        import geomagic.vendor.geomagic_core.pose_mapping as vendor_mapping
        import geomagic.vendor.geomagic_core.frames as vendor_frames
        for module in (vendor_rotations, vendor_controller, vendor_mapping, vendor_frames):
            profile.instrument(module, {'validate_rotation_matrix':'rotation.matrix_validation'})
        profile.instrument(vendor_mapping, {'matrix_to_rotation_vector':'rotation.so3_log',
                                            'rotation_vector_to_matrix':'rotation.so3_exp',
                                            'orthonormalize_rotation_matrix':'rotation.project_so3'})
    def elapsed():return 0. if state['start'] is None else time.monotonic()-state['start']
    def phase():
        t=elapsed()
        return 'idle' if t<idle_seconds else 'position' if t<idle_seconds+args.seconds_per_phase else 'rotation'
    class Socket:
        def setblocking(self,value):assert value is False
        def bind(self,address):pass
        def getsockname(self):return ('offline-memory',0)
        def close(self):pass
        def recvfrom(self,size):
            now=time.monotonic()
            if state['next_input'] is None:state['next_input']=now
            if now<state['next_input']:raise BlockingIOError
            t=state['next_input'];state['next_input']+=.002
            p=phase();angle=.3*math.sin(2*math.pi*.4*(t-(state['start'] or t)))
            vector=((0.,angle,0.) if args.rotation_mode=='rotation_z' else (angle,0.,0.)) if p=='rotation' else (0.,)*3
            magnitude=math.sqrt(sum(v*v for v in vector))
            factor=math.sin(magnitude/2)/magnitude if magnitude else 0.
            xyz=tuple(v*factor for v in vector)+(math.cos(magnitude/2),)
            pos=(.015*math.sin(2*math.pi*.4*(t-(state['start'] or t))),0.,0.) if p=='position' else (0.,)*3
            if args.sync:
                # Known SOFTWARE acquisition schedule; not a physical HDAPI
                # timestamp. Five finite 100 ms input ramps, 2.6 s holds.
                phase_t=(t-(state['start'] or t))-(idle_seconds if p=='position' else idle_seconds+args.seconds_per_phase if p=='rotation' else 0.)
                pulse=0.
                for step_index in range(5):
                    onset=.8+step_index*2.7
                    pulse+=(-1 if step_index%2 else 1)*max(0.,min(1.,(phase_t-onset)/.1))
                pos=(.02*pulse,0.,0.) if p=='position' else (.006*pulse,.004*pulse,.003*pulse) if p=='rotation' and args.rotation_mode=='6dof' else (0.,)*3
                vector=((0.,.2*pulse,0.) if args.rotation_mode=='rotation_z' else (.12*pulse,-.08*pulse,.14*pulse)) if p=='rotation' else (0.,)*3
                magnitude=math.sqrt(sum(v*v for v in vector));factor=math.sin(magnitude/2)/magnitude if magnitude else 0.
                xyz=tuple(v*factor for v in vector)+(math.cos(magnitude/2),)
            n=state['seq'];state['seq']+=1
            packet=GeomagicStatePacket(1,'offline-full-worker',n,t,'server.monotonic.synthetic' if args.sync else 'server.monotonic.hdapi',pos,xyz,p!='idle',False,True)
            return encode_packet(packet),('offline',0)
    sock=Socket()
    def initialize(b):
        with patch('geomagic.bridge.udp_state_client.socket.socket',return_value=sock):
            b.client=GeomagicUDPStateClient(('offline',0),max_packets_per_poll=256)
        if profile:
            profile.instrument(b.client,{'poll':'input.udp_receive_decode','latest_state':'input.fresh_state'})
        b.pipeline.mapper.activation_samples=2
    def runner_factory(*a,**kw):
        r=MimicLiveRunner(*a,**kw);state['runner']=r
        original=r.step_once
        previous=[time.monotonic()]
        if args.baseline_root and profile:
            b=r.backend
            profile.instrument(b.pipeline, {'update_mapping':'rotation.mapping_total'})
            profile.instrument(b.pipeline.mapper, {'update':'rotation.fresh_sample'})
            profile.instrument(b.pipeline.deadband, {'update':'rotation.deadband'})
            if b.tracking:profile.instrument(b.tracking, {'capture':'ui.diagnostics_stats'})
            profile.instrument(r.env, {'control':'simulation.env_control', 'step_physics':'simulation.mujoco_step'})
            b.profile=profile
        def step(**kw):
            if state['start'] is None:state['start']=time.monotonic()
            wall=time.monotonic();ph=phase();sim=float(r.data.time);calls=r.backend.control_updates
            if args.baseline_root and profile:
                profile.phase=ph
                profile.tick(wall-previous[0]);previous[0]=wall
            result=original(**kw)
            rows.append((wall,ph,float(r.data.time)-sim,r.backend.control_updates-calls,int(result[1] is not None),r.input_fresh))
            return result
        r.step_once=step
        return r
    camera_boundaries=[]
    def camera_boundary(label):
        if previews:
            p=previews[0]
            camera_boundaries.append((label,time.monotonic(),p.rendered,p.submitted,p.replaced,p.skipped))
    class Commands:
        def get_nowait(self):
            if state['start'] is not None and elapsed()>=idle_seconds+2*args.seconds_per_phase:
                if not state['done']:
                    state['done']=True
                    camera_boundary('end')
                    state['input']=state['runner'].input_metrics()
                    state['input']['accepted_wall_hz']=state['input'].get('packets_accepted',0)/elapsed()
                    return 'abort'
                raise queue.Empty
            ph=phase()
            if ph!=state['command_phase']:
                state['command_phase']=ph
                camera_boundary(ph)
                return 'teleop_mode:'+(args.rotation_mode if ph=='rotation' else 'position')
            raise queue.Empty
    class States:
        def put_nowait(self,payload):
            # Consume/serialize each real worker status; no accumulating frontend queue.
            payloads.append(json.loads(json.dumps(payload)))
        def get_nowait(self):raise queue.Empty
    previews=[];writers=[]
    def preview_factory(*a,**kw):
        kw['image_size']=tuple(args.camera_size);kw['operator_image_size']=tuple(args.camera_size)
        p=recording.AsyncLivePreview(*a,**kw);p.cancel=p.close;previews.append(p);return p
    def writer_factory(*a,**kw):
        w=recording.AsyncCameraWriter(*a,**kw);w.cancel=w.close;writers.append(w);return w
    with tempfile.TemporaryDirectory(prefix='geomagic_worker_bench_') as temp:
        config={'input_backend':'geomagic','geomagic_source':'synthetic_udp','task':'mimicgen_square',
                '_preview_dir':temp,'dataset_output_root':temp,'seed':0}
        # A baseline no-save case omits disk encoding but still uses the original
        # worker's observation/high-rate copies; this is noted in its report.
        if args.baseline_root and not args.record:
            class NoWriter:
                def submit(self,*a):pass
                def close(self):pass
                def cancel(self):pass
                def metrics(self):return {'baseline_no_writer':True}
            selected_writer=lambda *a,**kw:NoWriter()
        else:selected_writer=writer_factory
        with patch.object(GeomagicTeleopBackend,'initialize',initialize),patch.object(ui_worker,'MimicLiveRunner',runner_factory),patch.object(ui_worker,'AsyncLivePreview',preview_factory),patch.object(ui_worker,'AsyncCameraWriter',selected_writer):
            ui_worker.work_worker_main(config,Commands(),States())
    if args.baseline_root and profile:profile.write()
    errors=[p['message'] for p in payloads if p.get('phase')=='ERROR']
    results={'hardware':False,'browser':False,'real_worker':True,'diagnostics':args.diagnostics,'record':args.record,
             'sync_capture':str(args.output.with_suffix('.capture').resolve()) if args.sync else None,
             'rotation_mode':args.rotation_mode,
             'idle_seconds':idle_seconds,'active_phase_seconds':args.seconds_per_phase,
             'baseline_root':str(args.baseline_root) if args.baseline_root else None,'errors':errors,'phases':{},
             'cameras':previews[0].metrics() if previews else None,
             'recorder':writers[0].metrics() if writers else None,
             'input':state.get('input'),
             'baseline_note':'Original worker timing/mapper snapshot. No-save baseline still copies observations/high-rate diagnostics; no PNG writer.' if args.baseline_root else None}
    for ph in ('idle','position','rotation'):
        selected=[x for x in rows if x[1]==ph]
        if len(selected)<2:continue
        # Exclude the last step crossing a phase edge from dt-derived rates.
        duration=selected[-1][0]-selected[0][0]
        sample=selected[:-1];sim=sum(x[2] for x in sample);dls=sum(x[3] for x in sample)
        results['phases'][ph]={'wall_s':duration,'sim_s':sim,'physics_hz':len(sample)/duration,'dls_hz':dls/duration,
                              'rtf':sim/duration,'dataset_samples':sum(x[4] for x in selected)}
        results['phases'][ph]['input_not_fresh_ticks']=sum(not x[5] for x in selected)
        for a,b in zip(camera_boundaries,camera_boundaries[1:]):
            if a[0]==ph:
                wall=b[1]-a[1]
                results['phases'][ph]['preview']={
                    'wall_s':wall,'frames':b[2]-a[2],'produced_fps':(b[2]-a[2])/wall,
                    'requests':b[3]-a[3],'request_fps':(b[3]-a[3])/wall,
                    'replaced':b[4]-a[4],'skipped':b[5]-a[5]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
