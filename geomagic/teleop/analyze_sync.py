"""Offline raw-log synchronization analysis. Never opens a device or viewer.

python -B -m geomagic.teleop.analyze_sync CAPTURE_DIRECTORY --output REPORT_DIRECTORY
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from geomagic.teleop.input_deadband import DeadbandConfig
from geomagic.teleop.pose_mapper import POSITION_SCALE, ROTATION_SCALE, MODE_LABELS
from geomagic.teleop.sync_capture import CLOCK
from geomagic.vendor.geomagic_core.rotations import quaternion_xyzw_to_matrix, matrix_to_rotation_vector as Log


def stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return dict(count=0, median_ms=None, p95_ms=None, p99_ms=None, max_ms=None, std_ms=None)
    values *= 1000
    return dict(count=len(values), median_ms=float(np.median(values)), p95_ms=float(np.percentile(values,95)),
                p99_ms=float(np.percentile(values,99)), max_ms=float(np.max(values)), std_ms=float(np.std(values)))


def latency(end, start):
    if end is None or start is None or end < start:
        return None
    return (end-start)/1e9


def load(directory):
    records, metadata = [], []
    for path in sorted(Path(directory).glob('*.jsonl.gz')):
        with gzip.open(path,'rt') as stream:
            rows = [json.loads(line) for line in stream]
        if not rows or rows[0].get('clock') != CLOCK:
            raise ValueError(f'{path}: unsupported/missing monotonic clock domain')
        metadata.append(rows[0])
        records.extend(rows[1:])
    native, packets = [], []
    for path in sorted(Path(directory).glob('native-meta-*.json')):
        meta = json.loads(path.read_text())
        if meta.get('clock') != CLOCK:
            raise ValueError(f'{path}: native clock domain mismatch')
        metadata.append(meta)
        suffix = path.name.removeprefix('native-meta-').removesuffix('.json')
        for name, dest in (('servo',native),('packets',packets)):
            with (Path(directory)/f'native-{name}-{suffix}.csv').open() as stream:
                for raw in csv.DictReader(stream):
                    row = {}
                    for key,value in raw.items():
                        row[key] = (value if key=='session' else None if value=='' else
                                    float(value) if key in ('sim_time','fx','fy','fz','cfx','cfy','cfz','px','py','pz','qx','qy','qz','qw') else int(value))
                    dest.append(row)
    samples = {(r['session'],r['sample_index']):r for r in native if r['valid']}
    refined = {(p['session'],p['sequence']):samples.get((p['session'],p['sample_index'])) for p in packets}
    for row in records:
        source = refined.get((row.get('session'),row.get('sequence')))
        if source and row.get('T0_ns') is not None:
            row['T0_callback_ns'] = row['T0_ns']
            row['T0_ns'] = source['acquired_ns']
            row['T0_acquisition_refined'] = True
    return sorted(records,key=lambda r:r['monotonic_ns']), sorted(native,key=lambda r:r['monotonic_ns']), metadata


def projected(rows, field, baseline, direction, rotation=False):
    if rotation:
        return np.asarray([Log(np.reshape(r[field],(3,3)) @ baseline.T) @ direction for r in rows])
    return np.asarray([(np.asarray(r[field])-baseline) @ direction for r in rows])


def crossing(values, threshold, sustained=3):
    if len(values)<sustained:
        return None
    hits=np.flatnonzero(np.convolve((np.asarray(values)>=threshold).astype(int),np.ones(sustained,dtype=int),'valid')==sustained)
    return None if not len(hits) else int(hits[0])


def detect_steps(inputs, *, noise_s=.3, window_s=.02, quiet_s=.5, config=None):
    """Velocity over 20 ms, measured initial noise + six robust sigmas.

    Floor = existing device ON radius / 20 ms. Two accepted samples must
    exceed velocity threshold; a new trial requires 500 ms below threshold.
    The detected sample acquisition is T0, not a manually guessed timestamp.
    """
    config=config or DeadbandConfig()
    events=[]
    groups=[]
    for row in inputs:
        key=(row.get('session'),row.get('mode'))
        if not groups or groups[-1][0]!=key:
            groups.append((key,[]))
        groups[-1][1].append(row)
    for (session,mode),rows in groups:
        rows=[r for r in rows if r.get('T0_ns') is not None and r.get('valid')]
        if len(rows)<4:
            continue
        time_ns=np.array([r['T0_ns'] for r in rows],dtype=np.int64)
        if np.any(np.diff(time_ns)<=0):
            # Duplicate acquisition or a clock discontinuity cannot be motion.
            rows=[r for i,r in enumerate(rows) if i==0 or r['T0_ns']>rows[i-1]['T0_ns']]
            time_ns=np.array([r['T0_ns'] for r in rows],dtype=np.int64)
        positions=np.array([r['device_position'] for r in rows])
        rotations=[quaternion_xyzw_to_matrix(r['device_quaternion']) for r in rows]
        vp=np.zeros(len(rows));vr=np.zeros(len(rows));anchors=np.zeros(len(rows),dtype=int)
        for i in range(1,len(rows)):
            j=max(0,int(np.searchsorted(time_ns,time_ns[i]-window_s*1e9,side='right'))-1)
            anchors[i]=j
            dt=(time_ns[i]-time_ns[j])/1e9
            if dt>0:
                vp[i]=np.linalg.norm(positions[i]-positions[j])/dt
                vr[i]=np.linalg.norm(Log(rotations[i] @ rotations[j].T))/dt
        baseline=(time_ns-time_ns[0])/1e9<noise_s
        def threshold(values,floor):
            noise=values[baseline]
            center=float(np.median(noise))
            robust_sigma=1.4826*float(np.median(np.abs(noise-center)))
            return max(floor,center+6*robust_sigma)
        p_threshold=threshold(vp,config.translation_on_m/window_s)
        r_threshold=threshold(vr,config.rotation_on_rad/window_s)
        above=(vp>p_threshold) if mode=='position' else (vr>r_threshold) if mode in ('rotation_z','rotation') else ((vp>p_threshold)|(vr>r_threshold))
        moving=False;quiet_since=None;candidate=None
        for i,row in enumerate(rows):
            valid=row.get('grey') and not row.get('white') and not baseline[i]
            if not valid:
                moving=False;candidate=None;quiet_since=None
                continue
            if above[i]:
                quiet_since=None
                if not moving:
                    if candidate is None:
                        candidate=i
                    elif i==candidate+1:
                        start=rows[candidate]
                        events.append(dict(session=session,mode=mode,sequence=start['sequence'],T0_ns=start['T0_ns'],
                            baseline_input=rows[int(anchors[candidate])],
                            device_velocity_threshold_m_s=p_threshold,device_angular_threshold_rad_s=r_threshold))
                        moving=True;candidate=None
            else:
                candidate=None
                if quiet_since is None:quiet_since=time_ns[i]
                if (time_ns[i]-quiet_since)/1e9>=quiet_s:moving=False
    return events


def teleop_steps(records,config=None):
    config=config or DeadbandConfig()
    inputs=[r for r in records if r['kind']=='input']
    ticks=[r for r in records if r['kind']=='tick']
    frames=[r for r in records if r['kind']=='preview']
    detected=detect_steps(inputs,config=config)
    output=[]
    for index,event in enumerate(detected):
        t0=event['T0_ns']
        next_t0=next((e['T0_ns'] for e in detected[index+1:] if e['session']==event['session']),t0+3_000_000_000)
        end=min(next_t0,t0+3_000_000_000)
        selected=[r for r in ticks if r.get('session')==event['session'] and r.get('mode')==event['mode'] and t0<=r['monotonic_ns']<end]
        before=[r for r in ticks if r.get('session')==event['session'] and r.get('mode')==event['mode'] and t0-150_000_000<=r['monotonic_ns']<t0]
        if not selected or not before or not before[-1]['active']:
            continue
        event=dict(event, trial=len(output)+1, end_ns=end, channels={})
        baseline=before[-1]
        tail=selected[-max(1,len(selected)//5):]
        for channel,rotation,floor in (('position',False,POSITION_SCALE[0]*config.translation_off_m),
                                       ('rotation',True,ROTATION_SCALE*config.rotation_off_rad)):
            if event['mode']=='position' and rotation or event['mode'] in ('rotation_z','rotation') and not rotation:
                continue
            field='rotation' if rotation else 'position'
            raw_key='raw_'+field
            base=np.reshape(baseline[raw_key],(3,3)) if rotation else np.asarray(baseline[raw_key])
            vectors=np.asarray([Log(np.reshape(r[raw_key],(3,3))@base.T) if rotation else np.asarray(r[raw_key])-base for r in tail])
            final=np.median(vectors,axis=0)
            magnitude=float(np.linalg.norm(final))
            if magnitude<3*floor:
                continue
            direction=final/magnitude
            # One threshold per channel/trial, used for all three output signals.
            noise=max(float(np.std(projected(before,prefix+field,
                     np.reshape(baseline[prefix+field],(3,3)) if rotation else np.asarray(baseline[prefix+field]),
                     direction,rotation))) for prefix in ('raw_','conditioned_','actual_'))
            threshold=max(floor,6*noise)
            result=dict(threshold=threshold,requested_magnitude=magnitude,baseline=baseline, direction=direction.tolist())
            for prefix,stage in (('raw_','T2_ns'),('conditioned_','T3_ns'),('actual_','monotonic_ns')):
                b=np.reshape(baseline[prefix+field],(3,3)) if rotation else np.asarray(baseline[prefix+field])
                values=projected(selected,prefix+field,b,direction,rotation)
                first=crossing(values,threshold)
                stamp=None if first is None else selected[first].get(stage)
                result[prefix+'response_s']=latency(stamp,t0)
                result[prefix+'response_ns']=stamp
                if prefix=='actual_':
                    result['response_sim_time']=None if first is None else selected[first]['sim_time']
                    for fraction,label in ((.5,'t50_s'),(.9,'t90_s')):
                        hit=crossing(values,magnitude*fraction)
                        result[label]=None if hit is None else latency(selected[hit]['monotonic_ns'],t0)
                    result['final_tracking_fraction']=float(np.median(values[-max(1,len(values)//5):])/magnitude)
            event['channels'][channel]=result
        if not event['channels']:
            continue
        actuals=[c['actual_response_ns'] for c in event['channels'].values() if c['actual_response_ns'] is not None]
        event['T5_ns']=min(actuals) if actuals else None
        sims=[c['response_sim_time'] for c in event['channels'].values() if c['response_sim_time'] is not None]
        matched=[f for f in frames if sims and f['sim_time']>=min(sims)-1e-12 and f['monotonic_ns']>=(event['T5_ns'] or t0)
                 and f['monotonic_ns']<end]
        event['T6_ns']=matched[0]['monotonic_ns'] if matched else None
        event['visual_kind']=matched[0]['visual_kind'] if matched else None
        event['T4_first_generated_ns']=next((r['T4_ns'] for r in selected if r.get('T4_ns') is not None and r['T4_ns']>=t0),None)
        changed=[c['conditioned_response_ns'] for c in event['channels'].values() if c['conditioned_response_ns'] is not None]
        first_conditioned=min(changed) if changed else None
        event['T4_ns']=None
        if first_conditioned is not None:
            after=[r for r in selected if r.get('T4_ns') is not None and r['T4_ns']>=first_conditioned]
            if after and 'qtarget' in baseline:
                qbase=np.asarray(baseline['qtarget'])
                noise=float(np.std([np.linalg.norm(np.asarray(r['qtarget'])-qbase) for r in before]))
                threshold=max(1e-6,6*noise)
                hit=crossing([np.linalg.norm(np.asarray(r['qtarget'])-qbase) for r in after],threshold)
                if hit is not None:event['T4_ns']=after[hit]['T4_ns']
                event['joint_response_threshold_rad']=threshold
            elif after:
                event['T4_ns']=after[0]['T4_ns']
        output.append(event)
    return output


def haptics(records,native):
    commands=[r for r in records if r['kind']=='force' and r.get('sim_time') is not None]
    applied=[r for r in native if r.get('valid') and r.get('H4_ns')]
    frames=[r for r in records if r['kind']=='preview']
    joined={}
    for r in applied:
        key=(r['session'].removeprefix('teleop-geo-'),r['sequence'])
        joined.setdefault(key,r)
    transport=[]
    for c in commands:
        servo=joined.get((c['force_session'],c['force_sequence']))
        if servo:
            servo['sim_time']=c['sim_time']
            transport.append(latency(servo['H3_ns'],c['H2_ns']))
    events=[];release=[];previous=None
    for i,c in enumerate(commands):
        if previous is None:
            previous=c
            continue
        if c.get('contact') and not previous.get('contact'):
            end=next((r['H0_ns'] for r in commands[i+1:] if not r.get('contact')),c['H0_ns']+5_000_000_000)
            candidates=[r for r in commands[i:] if r.get('H2_ns',end)>=c['H0_ns'] and r.get('H2_ns',end)<end
                        and r.get('sent') and np.linalg.norm(r.get('device_force',(0,0,0)))>1e-9]
            hits=[]
            for command in candidates:
                servo=joined.get((command['force_session'],command['force_sequence']))
                if servo and np.linalg.norm([servo[k] for k in ('fx','fy','fz')])>1e-9:
                    hits.append((servo,command))
            first=min(hits,key=lambda x:x[0]['H4_ns']) if hits else None
            visible=next((r for r in frames if r['sim_time']>=c['sim_time']-1e-12 and r['monotonic_ns']>=c['H0_ns'] and r['monotonic_ns']<end),None)
            events.append(dict(H0_ns=c['H0_ns'],H1_ns=c.get('H1_ns'),H2_ns=candidates[0]['H2_ns'] if candidates else None,
                H3_ns=first[0]['H3_ns'] if first else None,H4_ns=first[0]['H4_ns'] if first else None,
                visible_ns=visible['monotonic_ns'] if visible else None,sim_time=c['sim_time']))
        reason=('contact_off' if previous.get('contact') and not c.get('contact') else
                'grey_release' if previous.get('grey') and not c.get('grey') else
                'clutch' if not previous.get('white') and c.get('white') else
                'host_stale' if previous.get('fresh') and not c.get('fresh') else None)
        if reason and np.linalg.norm(previous.get('device_force',(0,0,0)))>1e-9:
            end=c['H0_ns']+100_000_000
            zeros=[r for r in commands[i:] if c['H0_ns']<=r['H2_ns']<end and r.get('sent') and np.linalg.norm(r.get('device_force',(0,0,0)))<1e-9]
            native_zero=[joined.get((r['force_session'],r['force_sequence'])) for r in zeros]
            native_zero=[r for r in native_zero if r is not None and np.linalg.norm([r[k] for k in ('fx','fy','fz')])<1e-9]
            # Native buttons may force zero BEFORE the host sends its zero.
            direct=[r for r in applied if c['H0_ns']<=r['H4_ns']<end and np.linalg.norm([r[k] for k in ('fx','fy','fz')])<1e-9]
            off=min(native_zero+direct,key=lambda r:r['H4_ns']) if native_zero or direct else None
            release.append(dict(event=reason,start_ns=c['H0_ns'],command_zero_ns=zeros[0]['H2_ns'] if zeros else None,
                                applied_zero_ns=off['H4_ns'] if off else None,origin='host_detection'))
        previous=c
    for a,b in zip(applied,applied[1:]):
        if np.linalg.norm([a[k] for k in ('fx','fy','fz')])<=1e-9:
            continue
        reason=('native_grey_release' if a['buttons']&1 and not b['buttons']&1 else
                'native_clutch' if not a['buttons']&2 and b['buttons']&2 else
                'stale_command' if not a['stale'] and b['stale'] else None)
        if reason:
            start=b['command_ns']+20_000_000 if reason=='stale_command' else b['acquired_ns']
            zero=b['H4_ns'] if np.linalg.norm([b[k] for k in ('fx','fy','fz')])<1e-9 else None
            release.append(dict(event=reason,start_ns=start,command_zero_ns=None,applied_zero_ns=zero,origin='native_sample_or_watchdog_deadline'))
    return events,release,transport


def rates(records,native,ticks=None,recording_enabled=None):
    ticks=[r for r in records if r['kind']=='tick'] if ticks is None else ticks
    if len(ticks)<2:
        return {}
    start,end=ticks[0]['monotonic_ns'],ticks[-1]['monotonic_ns'];duration=(end-start)/1e9
    def rate(kind):return sum(r['kind']==kind and start<=r['monotonic_ns']<=end for r in records)/duration
    updates=sum(max(0,b['control_updates']-a['control_updates']) for a,b in zip(ticks,ticks[1:]))
    sim=sum(max(0,b['sim_time']-a['sim_time']) for a,b in zip(ticks,ticks[1:]))
    selected=[r for r in native if start<=r['monotonic_ns']<=end]
    accepted=[r.get('udp_accepted') for r in ticks]
    return dict(wall_s=duration,physics_hz=(len(ticks)-1)/duration,dls_hz=updates/duration,rtf=sim/duration,
        udp_accepted_hz=(accepted[-1]-accepted[0])/duration if accepted[-1] is not None and accepted[0] is not None else None,
        mapped_input_hz=rate('input'),hdapi_sample_hz=len(selected)/duration if selected else None,
        haptic_servo_hz=sum(bool(r.get('H4_ns')) for r in selected)/duration if selected else None,
        preview_produced_fps=rate('preview'),
        dataset_write_hz=None if recording_enabled and not any(r['kind']=='dataset_write' for r in records) else rate('dataset_write'),
        recording_enabled=recording_enabled,
        dataset_writes_after_control=sum(r['kind']=='dataset_write' and r['monotonic_ns']>end for r in records))


def rendering_durations(records):
    """Queue, rendering and completed writes, independently of control latency."""
    endpoints={
        'preview submit→produced':('preview','monotonic_ns','submitted_ns'),
        'preview render/encode/write':('preview','monotonic_ns','render_begin_ns'),
        'dataset queue wait':('dataset_write','dequeued_ns','submitted_ns'),
        'dataset render two cameras':('dataset_write','rendered_ns','render_started_ns'),
        'dataset encode/write two PNGs':('dataset_write','monotonic_ns','rendered_ns'),
        'dataset submit→written':('dataset_write','monotonic_ns','submitted_ns'),
    }
    return {name:stats([latency(r.get(end),r.get(start)) for r in records if r['kind']==kind])
            for name,(kind,end,start) in endpoints.items()}


def analyze(records,native,metadata):
    stages={name:[] for name in ('T1-T0','T2-T1','T3-T2','T4-T3','T4-T0','T5-T0','T6-T0')}
    by_mode={mode:dict(qtarget=[],qtarget_response=[],actual=[],channels={}) for mode in MODE_LABELS}
    seen=set()
    for row in records:
        if row['kind']!='tick':continue
        for name in ('T1-T0','T2-T1','T3-T2','T4-T3','T4-T0'):
            end,start=name.split('-');end_ns=row.get(end+'_ns');start_ns=row.get(start+'_ns')
            key=(row.get('session'),row.get('sequence'),name,end_ns)
            if key in seen:continue
            seen.add(key)
            value=latency(end_ns,start_ns);stages[name].append(value)
            if name=='T4-T0' and row.get('mode') in by_mode:by_mode[row['mode']]['qtarget'].append(value)
    settings=next((m['settings'] for m in metadata if m.get('settings',{}).get('deadband')),None)
    config=DeadbandConfig(**settings['deadband']) if settings else DeadbandConfig()
    recording_enabled=next((m['settings']['recording_enabled'] for m in metadata
                            if 'recording_enabled' in m.get('settings',{})),None)
    steps=teleop_steps(records,config)
    for event in steps:
        stages['T5-T0'].append(latency(event['T5_ns'],event['T0_ns']))
        stages['T6-T0'].append(latency(event['T6_ns'],event['T0_ns']))
        by_mode[event['mode']]['actual'].append(latency(event['T5_ns'],event['T0_ns']))
        by_mode[event['mode']]['qtarget_response'].append(latency(event['T4_ns'],event['T0_ns']))
        for channel,value in event['channels'].items():
            bucket=by_mode[event['mode']]['channels'].setdefault(channel,dict(response=[],t50=[],t90=[],unreached_t90=0))
            bucket['response'].append(value['actual_response_s']);bucket['t50'].append(value['t50_s']);bucket['t90'].append(value['t90_s'])
            bucket['unreached_t90']+=value['t90_s'] is None
    contacts,releases,transport=haptics(records,native)
    haptic={name:stats([latency(e.get(name.split('-')[0]+'_ns'),e.get(name.split('-')[1]+'_ns')) for e in contacts])
            for name in ('H1-H0','H2-H0','H4-H0')}
    haptic['H3-H2']=stats(transport)
    released={}
    for e in releases:
        bucket=released.setdefault(e['event'],dict(command=[],applied=[]))
        bucket['command'].append(latency(e['command_zero_ns'],e['start_ns']))
        bucket['applied'].append(latency(e['applied_zero_ns'],e['start_ns']))
    alignment=[(e['H4_ns']-e['visible_ns'])/1e9 for e in contacts if e['H4_ns'] is not None and e['visible_ns'] is not None]
    return dict(metadata=metadata,teleop_latency={k:stats(v) for k,v in stages.items()},
        coverage=dict(inputs=sum(r['kind']=='input' for r in records),
                      native_refined_inputs=sum(r['kind']=='input' and r.get('T0_acquisition_refined',False) for r in records),
                      foreign_clock_inputs=sum(r['kind']=='input' and r.get('T0_ns') is None for r in records),
                      backend_consume_proxy_inputs=sum(r['kind']=='input' and not r.get('accepted_timestamp_measured',False) for r in records),
                      native_force_writes=sum(bool(r.get('H4_ns')) for r in native)),
        compute_durations={stage:stats([latency(end,start) for end,start in
                            {(r.get(stage+'_ns'),r.get(stage+'_begin_ns')) for r in records if r['kind']=='tick'
                             and r.get(stage+'_ns') is not None and r.get(stage+'_begin_ns') is not None}]) for stage in ('T3','T4')},
        mode_comparison={mode:dict(input_qtarget=stats(value['qtarget']),input_qtarget_response=stats(value['qtarget_response']),input_actual=stats(value['actual']),
            channels={c:dict(response=stats(v['response']),t50=stats(v['t50']),t90=stats(v['t90']),unreached_t90=v['unreached_t90'])
                      for c,v in value['channels'].items()}) for mode,value in by_mode.items()},
        haptic_latency=haptic,haptic_release={k:{field:stats(values) for field,values in v.items()} for k,v in released.items()},
        visual_latency=dict(input_preview=stats([latency(e['T6_ns'],e['T0_ns']) for e in steps]),
                            contact_visible=stats([latency(e['visible_ns'],e['H0_ns']) for e in contacts]),
                            matching='approximate: first produced/swapped snapshot with simulation time at/after response/contact; browser scanout unmeasured'),
        rendering_durations=rendering_durations(records),
        haptic_minus_visual=stats(alignment),runtime=rates(records,native,recording_enabled=recording_enabled),steps=steps,contacts=contacts,releases=releases,
        runtime_by_mode={mode:rates(records,native,[r for r in records if r['kind']=='tick' and r.get('mode')==mode and r.get('active')],recording_enabled) for mode in MODE_LABELS},
        detection=dict(noise_window_s=.3,velocity_window_s=.02,quiet_interval_s=.5,
            onset='two consecutive velocity crossings; existing device ON/20ms floor and measured median+6 robust sigmas',
            response='3 consecutive projected samples above max(mapped OFF radius,6 baseline sigmas), shared across raw/conditioned/actual',
            rise='time from detected movement acquisition to 50%/90% of final raw requested pose; unreached values are null',
            command='per-update T4 generation may precede a change in filtered intent; separate qtarget_response requires a changed conditioned pose and 3 joint-command samples above max(1e-6 rad,6 baseline sigmas)',
            plot_markers='T2/T3/T4/T5 mark significant signal response to this step; per-update computation endpoints are reported separately'))


def plots(records,native,result,directory,max_trials=40):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    ticks=[r for r in records if r['kind']=='tick'];inputs=[r for r in records if r['kind']=='input']
    for event in result['steps'][:max_trials]:
        t0=event['T0_ns'];limit=event['end_ns']
        rows=[r for r in ticks if r.get('session')==event['session'] and r.get('mode')==event['mode'] and t0-100_000_000<=r['monotonic_ns']<limit]
        samples=[r for r in inputs if r.get('session')==event['session'] and r.get('mode')==event['mode'] and t0-100_000_000<=r['monotonic_ns']<limit]
        fig,axes=plt.subplots(len(event['channels']),1,figsize=(10,3.5*len(event['channels'])),squeeze=False)
        for ax,(channel,c) in zip(axes[:,0],event['channels'].items()):
            rotational=channel=='rotation';factor=180/np.pi if rotational else 1000
            field='rotation' if rotational else 'position';b=c['baseline'];direction=np.asarray(c['direction'])
            for prefix,label in (('raw_','raw target'),('conditioned_','conditioned target'),('actual_','actual EE')):
                base=np.reshape(b[prefix+field],(3,3)) if rotational else np.asarray(b[prefix+field])
                ax.plot([(r['monotonic_ns']-t0)/1e9 for r in rows],projected(rows,prefix+field,base,direction,rotational)*factor,label=label)
            device_base=event['baseline_input'];dp=np.asarray(device_base['device_position']);dr=quaternion_xyzw_to_matrix(device_base['device_quaternion'])
            device=[np.linalg.norm(Log(quaternion_xyzw_to_matrix(r['device_quaternion'])@dr.T)) if rotational else np.linalg.norm(np.asarray(r['device_position'])-dp) for r in samples]
            ax.plot([(r['monotonic_ns']-t0)/1e9 for r in samples],np.asarray(device)*factor,':',label='device displacement/angle')
            marks={'T0':t0,'T2':c.get('raw_response_ns'),'T3':c.get('conditioned_response_ns'),'T4':event.get('T4_ns'),'T5':c.get('actual_response_ns')}
            for number,(label,stamp) in enumerate(marks.items()):
                if stamp is not None:
                    x=(stamp-t0)/1e9;ax.axvline(x,color='grey',alpha=.5);ax.text(x,.98-number*.08,label,transform=ax.get_xaxis_transform())
            ax.axhline(c['threshold']*factor,color='grey',linestyle=':',label='response threshold')
            ax.set_ylabel('angle (deg)' if rotational else 'displacement (mm)');ax.set_xlabel('monotonic seconds from detected acquisition');ax.legend(loc='lower right')
        fig.suptitle(f"{MODE_LABELS[event['mode']]} trial {event['trial']}");fig.tight_layout();fig.savefig(root/f"teleop-{event['trial']:02d}.png",dpi=150);plt.close(fig)
    force=[r for r in records if r['kind']=='force' and r.get('sim_time') is not None]
    if force:
        first=force[0]['monotonic_ns'];fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True)
        t=[(r['monotonic_ns']-first)/1e9 for r in force]
        axes[0].step(t,[r.get('contact',False) for r in force],where='post');axes[0].set_ylabel('contact')
        for ax,field,label in ((axes[1],'world_force','world force (N)'),(axes[2],'device_force','device command (N)')):
            vectors=np.asarray([r.get(field,(0,0,0)) for r in force]);ax.plot(t,vectors,label=['X','Y','Z']);ax.set_ylabel(label);ax.legend()
        servo=[r for r in native if r.get('H4_ns')]
        if servo:axes[3].plot([(r['H4_ns']-first)/1e9 for r in servo],[[r[k] for k in ('fx','fy','fz')] for r in servo],label=['X','Y','Z']);axes[3].legend()
        else:axes[3].text(.02,.5,'HDAPI force application unavailable',transform=axes[3].transAxes)
        axes[3].set_ylabel('HDAPI write (N)');axes[3].set_xlabel('aligned monotonic seconds')
        for event in result['contacts'][:max_trials]:
            for stage in ('H0','H1','H2','H3','H4'):
                stamp=event.get(stage+'_ns')
                if stamp is not None:
                    x=(stamp-first)/1e9;axes[0].axvline(x,alpha=.4);axes[0].annotate(stage,(x,1),rotation=90)
        fig.tight_layout();fig.savefig(root/'haptic-sync.png',dpi=150);plt.close(fig)


def table(metrics):
    def cell(v):return 'N/A' if v is None else f'{v:.3f}'
    rows=['| Stage | Median ms | p95 ms | p99 ms | Max ms | Std ms | n |','|---|---:|---:|---:|---:|---:|---:|']
    for name,value in metrics.items():rows.append('| '+name+' | '+' | '.join(cell(value[k]) for k in ('median_ms','p95_ms','p99_ms','max_ms','std_ms'))+f" | {value['count']} |")
    return '\n'.join(rows)


def write_report(result,path):
    parts=['# Synchronization measurement','All latency uses CLOCK_MONOTONIC. Simulation time is used only to match snapshots. N/A means no valid matching measurement; it is never zero.',
        '# Teleop latency',table(result['teleop_latency']),'# Teleop response',
        '| Mode / channel | Response median ms | t50 median ms | t90 median ms | Unreached t90 |','|---|---:|---:|---:|---:|']
    def cell(v):return 'N/A' if v is None else f'{v:.3f}'
    for mode,data in result['mode_comparison'].items():
        for channel,c in data['channels'].items():parts.append(f"| {MODE_LABELS[mode]} / {channel} | {cell(c['response']['median_ms'])} | {cell(c['t50']['median_ms'])} | {cell(c['t90']['median_ms'])} | {c['unreached_t90']} |")
    parts += ['# Haptic latency',table(result['haptic_latency']),'# Haptic release latency']
    parts += [table({event+' / '+kind:values for event,data in result['haptic_release'].items() for kind,values in data.items()}),
        '# Visual latency',table({k:v for k,v in result['visual_latency'].items() if isinstance(v,dict)}),result['visual_latency']['matching'],
        '# Haptic/visual alignment',table({'haptic minus produced visual (approximate)':result['haptic_minus_visual']}),
        '# Mode comparison',table({MODE_LABELS[m]+' '+{'input_qtarget':'input→generated qtarget','input_qtarget_response':'input→changed qtarget','input_actual':'input→actual response'}[kind]:values for m,data in result['mode_comparison'].items() for kind,values in data.items() if kind!='channels'}),
        '# Rendering and recording',table(result['rendering_durations']),
        '# Runtime','```json',json.dumps(result['runtime'],indent=2),'```','# Detection method','```json',json.dumps(result['detection'],indent=2),'```',
        '# Limits','T0 is refined with the native acquisition stamp when available; otherwise it is the existing producer callback/synthetic timestamp. H4 is the successful HD_CURRENT_FORCE SDK write, with EndFrame commit also retained; physical force sensation is not measured. Visual matching is approximate and excludes browser/network/scanout delay. Rise times include the deliberate input ramp, deadband, conditioning and plant response; they are separate from stage latency. No optimization is inferred from unavailable physical data.']
    Path(path).write_text('\n\n'.join(parts).replace('|\n\n|','|\n|')+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('capture',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-plots',type=int,default=40);ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args()
    if not 0<=args.max_plots<=100:ap.error('max-plots must be in [0,100]')
    records,native,metadata=load(args.capture);result=analyze(records,native,metadata)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_report(result,args.output/'analysis.md')
    if not args.no_plots:plots(records,native,result,args.output/'plots',args.max_plots)
    print(json.dumps(dict(runtime=result['runtime'],detected_steps=len(result['steps']),contacts=len(result['contacts'])),indent=2))


if __name__=='__main__':main()
