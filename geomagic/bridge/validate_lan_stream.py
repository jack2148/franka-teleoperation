"""Explicit-address LAN receiver -> existing pose mapper -> bounded metrics only.

No FR3 controller, MuJoCo, hardware, UI or recording imports. Start manually.
Receive timestamps are userspace dequeue times, not kernel ingress timestamps.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from geomagic.bridge.synthetic_sender import ipv4_host
from geomagic.bridge.timing_stats import TimingStats, positive_number
from geomagic.bridge.udp_state_client import GeomagicUDPStateClient
from geomagic.teleop.pose_mapper import GeomagicPoseMapper
from geomagic.vendor.geomagic_core.rotations import matrix_to_rotation_vector

GEOMAGIC_ROOT = Path(__file__).resolve().parents[1]


class StreamMetrics:
    def __init__(self, *, stale_timeout=.05, event_capacity=256):
        self.stale_timeout = stale_timeout
        self.event_capacity = event_capacity
        self.events_truncated = 0
        self.arrivals = TimingStats()
        self.same_session_arrivals = TimingStats()
        self.poll_intervals = TimingStats()
        self.first_receive = self.last_receive = None
        self.last_sequence = self.session = None
        self.session_segments = 0
        self.missing = self.accepted = self.sequence_violations = 0
        self.session_events = 0
        self.seen_fresh = False
        self.open_stale = None
        self.stale_periods = []
        self.stale_transitions = self.closed_stale_periods = 0
        self.max_receive_age_ms = None
        self.mapper_samples = self.reanchors = 0
        self.activations = []
        self.activation_count = 0
        self.gate_start = None
        self.gate_samples = 0
        self.previous_target = None
        self.max_position_step = self.max_rotation_step = 0.
        self.max_position_discontinuity = self.max_rotation_discontinuity = 0.
        self.max_reanchor_actual_position_error = self.max_reanchor_actual_rotation_error = 0.
        self.rejected_observations = self.rejected_refresh_violations = 0
        self.rejected_mapper_update_violations = 0

    def append(self, collection, event):
        if len(collection) < self.event_capacity:
            collection.append(event)
        else:
            self.events_truncated += 1

    def reset_gate(self):
        self.gate_start = None
        self.gate_samples = 0

    def freshness(self, now, status):
        if status.age_ms is not None:
            self.max_receive_age_ms = max(self.max_receive_age_ms or 0., status.age_ms)
        if status.fresh:
            self.seen_fresh = True
            if self.open_stale is not None:
                event = dict(self.open_stale, end_server_monotonic=now,
                             duration_s=max(0., now - self.open_stale["start_server_monotonic"]),
                             terminal_censored=False)
                self.append(self.stale_periods, event)
                self.closed_stale_periods += 1
                self.open_stale = None
        elif status.stale and self.seen_fresh and self.open_stale is None:
            # Estimate threshold crossing from the receiver's valid-receive age.
            start = now - max(0., (status.age_ms or 0.) / 1000. - self.stale_timeout)
            self.open_stale = dict(start_server_monotonic=start,
                                   first_observed_server_monotonic=now)
            self.stale_transitions += 1
            self.reset_gate()

    def accept(self, state):
        self.accepted += 1
        if self.first_receive is None:
            self.first_receive = state.timestamp
        if self.last_receive is not None:
            gap = max(0., state.timestamp - self.last_receive) * 1000.
            self.arrivals.add(gap)
            if state.session_id == self.session:
                self.same_session_arrivals.add(gap)
        if state.session_id != self.session:
            self.session_segments += 1
            # Synthetic runs start at zero; first received seq reveals prefix loss.
            self.missing += state.sequence
            self.reset_gate()
        else:
            difference = state.sequence - self.last_sequence
            if difference <= 0:
                self.sequence_violations += 1
            else:
                self.missing += difference - 1
        self.session, self.last_sequence = state.session_id, state.sequence
        self.last_receive = state.timestamp

    def mapping(self, target, state, session_changed, *, actual_position, actual_rotation):
        if session_changed:
            self.session_events += 1
            self.reset_gate()
        if not target.fresh:
            self.reset_gate()
        if target.mapper_updated:
            self.mapper_samples += 1
            self.gate_samples += 1
            if self.gate_start is None:
                self.gate_start = state.timestamp
        if target.reanchored:
            self.reanchors += 1
            if target.reason == "activated":
                self.activation_count += 1
                elapsed = max(0., state.timestamp - self.gate_start)
                self.append(self.activations, dict(
                    session_id=state.session_id, sequence=state.sequence,
                    elapsed_ms=elapsed * 1000., sample_count=self.gate_samples,
                    sample_span_rate_hz=(self.gate_samples - 1) / elapsed if elapsed else None,
                ))
        if target.synchronized or target.reanchored:
            self.max_reanchor_actual_position_error = max(
                self.max_reanchor_actual_position_error,
                float(np.linalg.norm(target.position - actual_position)))
            self.max_reanchor_actual_rotation_error = max(
                self.max_reanchor_actual_rotation_error,
                float(np.linalg.norm(matrix_to_rotation_vector(target.rotation @ actual_rotation.T))))
        if self.previous_target is not None:
            dp = float(np.linalg.norm(target.position - self.previous_target.position))
            dr = float(np.linalg.norm(matrix_to_rotation_vector(
                target.rotation @ self.previous_target.rotation.T)))
            self.max_position_step = max(self.max_position_step, dp)
            self.max_rotation_step = max(self.max_rotation_step, dr)
            # Normal commanded motion is NOT a reanchor discontinuity.
            if target.reanchored or not target.active or target.clutched:
                self.max_position_discontinuity = max(self.max_position_discontinuity, dp)
                self.max_rotation_discontinuity = max(self.max_rotation_discontinuity, dr)
        self.previous_target = target

    def rejected(self, before_state, after_state, before_updates, after_updates):
        self.rejected_observations += 1
        before_key = None if before_state is None else (
            before_state.session_id, before_state.sequence, before_state.timestamp)
        after_key = None if after_state is None else (
            after_state.session_id, after_state.sequence, after_state.timestamp)
        self.rejected_refresh_violations += before_key != after_key
        self.rejected_mapper_update_violations += before_updates != after_updates

    def report(self, now, status):
        periods = list(self.stale_periods)
        if self.open_stale is not None:
            self.append(periods, dict(self.open_stale, end_server_monotonic=now,
                        duration_s=max(0., now - self.open_stale["start_server_monotonic"]),
                        terminal_censored=True))
        span = None if self.first_receive is None else self.last_receive - self.first_receive
        timing = self.arrivals.report()
        denom = self.accepted + self.missing
        return dict(
            transport=asdict(status),
            timing=dict(accepted_rate_hz=(self.accepted - 1) / span if span and self.accepted > 1 else None,
                        receive_span_s=span, inter_arrival_ms=timing,
                        same_session_inter_arrival_ms=self.same_session_arrivals.report(),
                        poll_interval_ms=self.poll_intervals.report(),
                        timestamp_basis="server monotonic userspace recvfrom; not kernel ingress or one-way latency"),
            freshness=dict(stale_transitions=self.stale_transitions,
                           completed_stale_periods=self.closed_stale_periods,
                           stale_periods=periods, max_receive_age_ms=self.max_receive_age_ms,
                           note="startup wait excluded; terminal_censored includes expected sender shutdown silence"),
            sequence=dict(missing_count=self.missing, estimated_loss_fraction=self.missing / denom if denom else None,
                          session_segments=self.session_segments, accepted_sequence_violations=self.sequence_violations,
                          note="assumes each unique synthetic session starts at zero; unobserved tail loss is unknown"),
            mapper=dict(distinct_samples_processed=self.mapper_samples,
                        activation_count=self.activation_count, activations=self.activations,
                        reanchors=self.reanchors, session_change_events=self.session_events,
                        max_position_target_discontinuity_m=self.max_position_discontinuity,
                        max_rotation_target_discontinuity_rad=self.max_rotation_discontinuity,
                        max_reanchor_actual_position_error_m=self.max_reanchor_actual_position_error,
                        max_reanchor_actual_rotation_error_rad=self.max_reanchor_actual_rotation_error,
                        max_normal_or_reanchor_step_m=self.max_position_step,
                        max_normal_or_reanchor_step_rad=self.max_rotation_step),
            rejection_checks=dict(observations=self.rejected_observations,
                                  receive_timestamp_refresh_violations=self.rejected_refresh_violations,
                                  mapper_update_violations=self.rejected_mapper_update_violations),
            events_truncated=self.events_truncated,
            practical_checks=dict(
                sufficient_samples=self.accepted >= 50,
                zero_invalid=status.invalid_packets == 0 and status.unsupported_version_packets == 0,
                zero_duplicate_or_reordered_accepted=self.sequence_violations == 0,
                zero_completed_stale_periods=self.closed_stale_periods == 0,
                p99_below_25ms=timing["p99"] is not None and timing["p99"] < 25.,
                maximum_gap_below_50ms=timing["maximum"] is not None and timing["maximum"] < 50.,
                reanchor_uses_actual=self.max_reanchor_actual_position_error <= 1e-12 and self.max_reanchor_actual_rotation_error <= 1e-12,
                note="individual evidence, not automatic LAN PASS; inspect sender counts, gaps, terminal tail and experiment type"),
        )


def monitor(client, *, duration=60., wait_timeout=30., tail=.2, mode="position",
            clock=time.monotonic, sleep=time.sleep):
    """One-datagram poll budget preserves every accepted observation.

    Existing default client behavior is unchanged. This metrics runner deliberately
    processes every accepted packet instead of coalescing a whole drain to one sample.
    """
    mapper = GeomagicPoseMapper(np.zeros(3), np.eye(3), mode=mode)
    metrics = StreamMetrics(stale_timeout=client.stale_timeout_s)
    started = last_tick = clock()
    interrupted = False
    while True:
        try:
            now = clock()
            metrics.poll_intervals.add(max(0., now - last_tick) * 1000.)
            last_tick = now
            before_status, before_state = client.status(), client.latest_state()
            metrics.freshness(now, before_status)
            updates = mapper.mapper_updates
            accepted = client.poll()
            now = clock()
            status, state = client.status(), client.latest_state()
            if accepted is not None:
                metrics.accept(accepted)
            metrics.freshness(now, status)
            changed = status.session_changed
            target = mapper.update(state, now=now, actual_position=np.zeros(3),
                                   actual_rotation=np.eye(3), session_changed=changed)
            metrics.mapping(target, state, changed, actual_position=np.zeros(3), actual_rotation=np.eye(3))
            if changed:
                client.acknowledge_session_change()
            if status.packets_received > before_status.packets_received and accepted is None:
                metrics.rejected(before_state, state, updates, mapper.mapper_updates)
            if metrics.first_receive is None:
                if now - started >= wait_timeout:
                    reason = "no_accepted_state_before_wait_timeout"
                    break
            elif now - metrics.first_receive >= duration + tail:
                reason = "measurement_window_complete"
                break
            if status.packets_received == before_status.packets_received:
                sleep(.001)
        except KeyboardInterrupt:
            interrupted = True
            reason = "interrupted"
            break
    now = clock()
    result = metrics.report(now, client.status())
    result.update(kind="lan_receiver_metrics_only", stop_reason=reason,
                  requested_window_s=duration, requested_tail_s=tail,
                  elapsed_runtime_s=now - started, mode=mode, interrupted=interrupted,
                  activation_required_samples=50, stale_timeout_s=client.stale_timeout_s)
    return result


def output_path(text):
    path = Path(text).resolve()
    if not path.is_relative_to(GEOMAGIC_ROOT) or path == GEOMAGIC_ROOT:
        raise ValueError("reports must be saved under geomagic/")
    if path.exists():
        raise ValueError("report already exists; choose a new filename")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-address", required=True, type=ipv4_host)
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--duration", type=positive_number, default=60.)
    parser.add_argument("--wait-timeout", type=positive_number, default=30.)
    parser.add_argument("--tail", type=positive_number, default=.2)
    parser.add_argument("--mode", choices=("position", "rotation"), default="position")
    parser.add_argument("--output", type=output_path)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if max(args.duration, args.wait_timeout, args.tail) > 3600:
        parser.error("each time limit must be <= 3600 s")
    try:
        with GeomagicUDPStateClient((args.bind_address, args.port), max_packets_per_poll=1) as client:
            print(f"Listening on {client.address}; metrics only. Start the sender manually.", flush=True)
            report = monitor(client, duration=args.duration, wait_timeout=args.wait_timeout,
                             tail=args.tail, mode=args.mode)
        payload = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                output.write(payload + "\n")
        print(payload)
    except OSError as error:
        parser.exit(2, f"Validation stopped: {error}\nNo firewall/network changes attempted.\n")
    if report["stop_reason"] == "no_accepted_state_before_wait_timeout":
        print("No accepted state: inspect received/invalid counters and the manually entered endpoint. If host/network configuration blocks traffic, stop and report it; do not change firewall rules.")
        return 2
    return 130 if report["interrupted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
