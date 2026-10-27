"""Opt-in bounded profiler for the real worker and its camera processes.

Inclusive timings overlap. No profile files/percentiles are computed on physics
steps; reports are written at shutdown. All clocks use server monotonic time.
"""
from collections import defaultdict, deque
import functools
import json
import os
from pathlib import Path
import time
import threading

import numpy as np


class RuntimeProfiler:
    def __init__(self, *, capacity=4096, report_path=None):
        self._lock = threading.Lock()
        self.capacity = capacity
        self.report_path = report_path
        self.phase = 'idle'
        self.stats = {}
        self.wall = defaultdict(float)
        self.current = None
        self.last_qtarget_input_time = None
        self.events = deque(maxlen=128)
        self.pending_responses = deque(maxlen=128)
        self.last_sequence = None

    def record(self, name, seconds, *, phase=None):
        with self._lock:
            key = f'{phase or self.phase}.{name}'
            s = self.stats.get(key)
            if s is None:
                s = self.stats[key] = [0, 0., 0., deque(maxlen=self.capacity)]
            s[0] += 1; s[1] += seconds; s[2] = max(s[2], seconds); s[3].append(seconds)

    def wrap(self, name, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            start = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                self.record(name, time.monotonic() - start)
        return measured

    def instrument(self, obj, names):
        for attribute, name in names.items():
            setattr(obj, attribute, self.wrap(name, getattr(obj, attribute)))

    def wrap_lock(self, name, lock):
        """Measure acquisition of an existing lock without changing its owner."""
        profiler = self
        class MeasuredLock:
            def __enter__(self):
                start = time.monotonic()
                lock.acquire()
                profiler.record(name, time.monotonic() - start)
                return self

            def __exit__(self, *exception):
                lock.release()
        return MeasuredLock()

    def tick(self, elapsed):
        self.wall[self.phase] += elapsed
        self.record('scheduler.tick', elapsed)

    def sample(self, state):
        if state is None or (state.session_id, state.sequence) == self.last_sequence:
            return
        self.last_sequence = state.session_id, state.sequence
        t0 = state.source_timestamp if state.source_clock_domain == 'server.monotonic.hdapi' else None
        # Other clocks have no known offset; do not invent an input latency.
        self.current = {'session': state.session_id, 'sequence': state.sequence,
                        'T0': t0, 'T1': state.timestamp}
        self.events.append(self.current)
        if t0 is not None and 0 <= state.timestamp - t0 < 60:
            self.record('latency.input_udp', state.timestamp - t0)

    def stage(self, name):
        if self.current is None or name in self.current:
            return
        now = time.monotonic()
        self.current[name] = now
        t0 = self.current['T0']
        if t0 is not None and 0 <= now - t0 < 60:
            self.record('latency.input_' + {'T2':'mapper', 'T3':'conditioned', 'T4':'qtarget'}[name], now - t0)
        if name == 'T4':
            self.last_qtarget_input_time = t0

    def arm_response(self, qpos):
        if self.current is not None and 'T4' in self.current and 'q_before' not in self.current:
            self.current['q_before'] = qpos.copy()
            self.pending_responses.append(self.current)

    def response(self, qpos):
        # Approximate matching: first >1e-6 rad joint change after qtarget.
        # Existing velocity/contact can also cause it; not physical causality.
        retained = deque(maxlen=128)
        for e in self.pending_responses:
            if np.max(np.abs(qpos - e['q_before'])) > 1e-6:
                e['T5'] = time.monotonic()
                if e['T0'] is not None and 0 <= e['T5'] - e['T0'] < 60:
                    self.record('latency.input_sim_response_approx', e['T5'] - e['T0'])
            elif time.monotonic() - e['T4'] < .5:
                retained.append(e)
        self.pending_responses = retained

    def summary(self):
        result = {}
        with self._lock:
            snapshot = [(key, (s[0],s[1],s[2],list(s[3]))) for key,s in self.stats.items()]
        for key, s in snapshot:
            phase = key.split('.')[0]; values = np.asarray(s[3]) * 1000
            result[key] = {'calls': s[0], 'mean_ms': s[1]/s[0]*1000,
                           'p50_ms': float(np.percentile(values, 50)),
                           'p95_ms': float(np.percentile(values, 95)),
                           'p99_ms': float(np.percentile(values, 99)),
                           'max_ms': s[2]*1000,
                           'calls_per_wall_s': s[0]/self.wall[phase] if self.wall[phase] else None}
        return {'pid': os.getpid(), 'window_capacity': self.capacity,
                'phase_wall_s': dict(self.wall), 'timings': result,
                'latency_matching': 'T0 HDAPI monotonic; T1 receive; T2 mapper; T3 conditioner; T4 qtarget. T5 first joint change >1e-6 rad after command, approximate, may include earlier momentum/contact. T6 rendered latest state with qtarget input timestamp; not guaranteed causal response.',
                'sample_events_recent': [{k:v for k,v in e.items() if k != 'q_before'} for e in self.events]}

    def write(self, suffix=''):
        if not self.report_path:
            return
        p = Path(self.report_path + suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.summary(), indent=2, allow_nan=False) + '\n')


_INSTANCE = None

def get_profiler():
    global _INSTANCE
    if _INSTANCE is None and os.environ.get('MIMIC_GEOMAGIC_PROFILE') == '1':
        _INSTANCE = RuntimeProfiler(report_path=os.environ.get('MIMIC_GEOMAGIC_PROFILE_REPORT'))
    return _INSTANCE
