"""Bounded raw synchronization capture; serialize after control stops.

GEOMAGIC_SYNC_CAPTURE is an explicit output directory. Clocks are Linux
CLOCK_MONOTONIC nanoseconds. Simulation time is a separate column, never a
latency clock. Unknown simulation/session context is null until offline join.
"""
import atexit
import gzip
import json
import math
import os
from pathlib import Path
import time

CLOCK = 'linux.CLOCK_MONOTONIC'
LOCAL_SOURCE_CLOCKS = ('server.monotonic.hdapi', 'server.monotonic.synthetic')


def ns(seconds):
    return None if seconds is None else round(seconds * 1e9)


class SyncCapture:
    def __init__(self, directory, *, seconds=60., capacity=200000, clock=time.monotonic_ns):
        if not math.isfinite(seconds) or not 0 < seconds <= 600:
            raise ValueError('sync capture seconds must be in (0, 600]')
        if type(capacity) is not int or not 1 <= capacity <= 500000:
            raise ValueError('sync capture capacity must be in [1, 500000]')
        self.directory = Path(directory)
        self.clock = clock
        self.seconds, self.capacity = seconds, capacity
        self.started_ns = clock()
        self.first_record_ns = None
        self.records = []
        self.current = None
        self.last_input = None
        self.stopped = None
        self.written = False
        self.pid = os.getpid()
        self.role = 'host'
        self.settings = {}

    def emit(self, kind, sim_time=None, *, timestamp_ns=None, session=None, sequence=None, **fields):
        stamp = self.clock() if timestamp_ns is None else timestamp_ns
        if self.stopped is not None:
            return
        if self.first_record_ns is None:
            self.first_record_ns = stamp
        if stamp - self.first_record_ns > self.seconds * 1e9:
            self.stopped = 'duration_limit'
            return
        if len(self.records) >= self.capacity:
            self.stopped = 'record_limit'
            return
        self.records.append(dict(kind=kind, monotonic_ns=stamp, sim_time=sim_time,
                                 session=session, sequence=sequence, **fields))

    def input(self, state, sim_time, *, accepted_ns=None, mode=None):
        if state is None:
            self.current = None
            return
        key = state.session_id, state.sequence
        if key == self.last_input:
            return
        self.last_input = key
        self.current = dict(session=state.session_id, sequence=state.sequence,
                            T0_ns=ns(state.source_timestamp) if state.source_clock_domain in LOCAL_SOURCE_CLOCKS else None,
                            T1_ns=accepted_ns if accepted_ns is not None else self.clock())
        self.emit('input', sim_time, **self.current,
                  source_clock=state.source_clock_domain,
                  device_position=list(state.position_xyz), device_quaternion=list(state.orientation_xyzw),
                  grey=state.grey_button, white=state.white_button, valid=state.valid,
                  accepted_timestamp_measured=accepted_ns is not None, mode=mode)

    def stage(self, name):
        if self.current is not None:
            self.current[name + '_ns'] = self.clock()

    def preview_context(self):
        return {} if self.current is None else {k:v for k,v in self.current.items() if k in ('session','sequence','T0_ns')}

    def tick(self, backend):
        if self.stopped is not None:
            return
        target = backend.pipeline.last_raw_target
        if target is None:
            return
        c = backend.pipeline.controller
        position, rotation = c.actual_pose()
        self.emit('tick', float(backend.data.time), **(self.current or {}),
                  mode=target.mode, fresh=target.fresh, active=backend.teleop_active,
                  clutched=target.clutched, reanchored=target.reanchored,
                  raw_position=target.position.tolist(), raw_rotation=target.rotation.reshape(9).tolist(),
                  filtered_position=backend.pipeline.last_target.position.tolist(),
                  filtered_rotation=backend.pipeline.last_target.rotation.reshape(9).tolist(),
                  conditioned_position=c.target_position.tolist(), conditioned_rotation=c.target_rotation.reshape(9).tolist(),
                  actual_position=position.tolist(), actual_rotation=rotation.reshape(9).tolist(),
                  qtarget=backend.hold_q_target.tolist(), control_updates=backend.control_updates,
                  udp_accepted=getattr(backend.client, '_counts', {}).get('packets_accepted'))

    def write(self):
        if self.written:
            return
        # Freeze now so offline dataset replay cannot enter the physical log.
        self.stopped = self.stopped or 'session_end'
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f'{self.role}-{self.pid}.jsonl.gz'
        temporary = destination.with_name(destination.name + '.tmp')
        with gzip.open(temporary, 'wt') as stream:
            stream.write(json.dumps(dict(kind='metadata', clock=CLOCK, monotonic_ns=self.started_ns,
                sim_time=None, session=None, sequence=None, role=self.role, pid=self.pid,
                seconds_limit=self.seconds, record_limit=self.capacity, records=len(self.records),
                settings=self.settings, schema_version=1,
                clock_scope='all producers/consumers must run on the same Linux host; remote clocks require explicit conversion',
                stopped=self.stopped or 'session_end',
                T0_fallback='existing producer callback timestamp; native log refines acquisition',
                T1_fallback='backend-consume proxy only when UDP acceptance instrumentation is unavailable; flagged per input',
                visual_matching='snapshot simulation time; produced/swap completion, not browser display'), allow_nan=False)+'\n')
            for record in self.records:
                stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
        temporary.replace(destination)
        self.written = True


_INSTANCE = None


def get_sync_capture():
    global _INSTANCE
    directory = os.environ.get('GEOMAGIC_SYNC_CAPTURE')
    if not directory:
        return None
    if _INSTANCE is None or _INSTANCE.pid != os.getpid():
        _INSTANCE = SyncCapture(directory,
            seconds=float(os.environ.get('GEOMAGIC_SYNC_SECONDS', '60')),
            capacity=int(os.environ.get('GEOMAGIC_SYNC_RECORDS', '200000')))
        atexit.register(_INSTANCE.write)
    return _INSTANCE
