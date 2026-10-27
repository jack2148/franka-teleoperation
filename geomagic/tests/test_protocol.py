"""Direct parser tests; no network, hardware, or external runtime required."""

from dataclasses import asdict, replace
import json

import pytest

from geomagic.bridge.protocol import (
    MAX_PACKET_BYTES, MAX_SEQUENCE, GeomagicStatePacket, ProtocolError,
    UnsupportedVersionError, decode_packet, encode_packet,
)
from geomagic.vendor.geomagic_core.state import GeomagicState


def example_packet():
    return GeomagicStatePacket(
        protocol_version=1, session_id="touch-pc:boot-001", sequence=12,
        source_timestamp=1234.5, source_clock_domain="touch-pc.monotonic",
        position_m=(0.012, -0.034, 0.056),
        orientation_xyzw=(0.0, 0.0, 0.6, 0.8),
        grey_button=True, white_button=False, valid=True,
    )


def wire(**changes):
    fields = asdict(example_packet())
    fields.update(changes)
    return json.dumps(fields).encode()


def test_packet_round_trip_and_deterministic_bytes():
    packet = example_packet()
    payload = encode_packet(packet)
    assert decode_packet(payload) == packet
    assert encode_packet(decode_packet(payload)) == payload
    assert payload == json.dumps(asdict(packet), sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode()


def test_pose_and_button_conventions_preserved():
    packet = decode_packet(encode_packet(example_packet()))
    assert packet.position_m == (0.012, -0.034, 0.056)
    assert packet.orientation_xyzw == (0.0, 0.0, 0.6, 0.8)
    assert packet.grey_button is True
    assert packet.white_button is False


@pytest.mark.parametrize("orientation", [
    (0.0, 0.0, 0.0, 0.96), (0.0, 0.0, 0.0, 1.04),
    (0.0, 0.0, -0.6, -0.8),
])
def test_reference_tolerance_preserves_quaternion_without_normalizing(orientation):
    assert decode_packet(wire(orientation_xyzw=orientation)).orientation_xyzw == orientation


@pytest.mark.parametrize("field,value", [
    ("position_m", [float("nan"), 0, 0]),
    ("position_m", [0, float("inf"), 0]),
    ("position_m", [0, 0, -float("inf")]),
    ("orientation_xyzw", [0, 0, 0, float("nan")]),
    ("orientation_xyzw", [0, 0, float("inf"), 1]),
    ("source_timestamp", float("inf")),
    ("position_m", [0, 0]),
    ("position_m", [0, 0, 0, 0]),
    ("position_m", "123"),
    ("position_m", ["0", 0, 0]),
    ("position_m", [True, 0, 0]),
    ("position_m", [None, 0, 0]),
    ("position_m", [10**400, 0, 0]),
    ("orientation_xyzw", None),
    ("orientation_xyzw", [0, 0, 1]),
    ("orientation_xyzw", [0, 0, 0, 0]),
    ("orientation_xyzw", [0, 0, 0, 2]),
    ("orientation_xyzw", [0, 0, 0, 1.051]),
    ("orientation_xyzw", [0, 0, 0, 0.949]),
    ("orientation_xyzw", [1e308, 1e308, 1e308, 1e308]),
    ("orientation_xyzw", [0, 0, 0, "1"]),
    ("sequence", True),
    ("sequence", -1),
    ("sequence", 1.0),
    ("sequence", "1"),
    ("sequence", MAX_SEQUENCE + 1),
    ("session_id", ""),
    ("session_id", " "),
    ("session_id", 12),
    ("session_id", "a" * 129),
    ("session_id", "boot\n1"),
    ("source_clock_domain", ""),
    ("source_clock_domain", None),
    ("source_clock_domain", "a" * 65),
    ("source_timestamp", True),
    ("source_timestamp", -0.1),
    ("source_timestamp", "1"),
    ("protocol_version", True),
    ("protocol_version", 1.0),
    ("grey_button", 1),
    ("white_button", "false"),
    ("valid", 0),
])
def test_invalid_fields_rejected(field, value):
    with pytest.raises(ProtocolError):
        decode_packet(wire(**{field: value}))


def test_unsupported_version_has_distinct_error():
    with pytest.raises(UnsupportedVersionError):
        decode_packet(wire(protocol_version=2))


@pytest.mark.parametrize("payload", [
    b"", b"not json", b"\xff", b"[]", b"null", b"{}", b"{} trailing",
    b" " * (MAX_PACKET_BYTES + 1), b"[" * 1500 + b"]" * 1500,
])
def test_malformed_json_rejected(payload):
    with pytest.raises(ProtocolError):
        decode_packet(payload)


def test_duplicate_json_keys_rejected():
    payload = wire().replace(b'"sequence": 12', b'"sequence": 12, "sequence": 13')
    with pytest.raises(ProtocolError, match="duplicate JSON key"):
        decode_packet(payload)


def test_missing_and_unknown_fields_rejected():
    fields = asdict(example_packet())
    del fields["valid"]
    with pytest.raises(ProtocolError):
        decode_packet(json.dumps(fields).encode())
    with pytest.raises(ProtocolError):
        decode_packet(wire(gripper=255))


def test_json_exponent_overflow_rejected():
    with pytest.raises(ProtocolError):
        decode_packet(wire().replace(b"1234.5", b"1e999"))


def test_invalid_source_flag_preserved_but_pose_still_validated():
    assert decode_packet(wire(valid=False)).valid is False
    with pytest.raises(ProtocolError):
        decode_packet(wire(valid=False, orientation_xyzw=[0, 0, 0, 0]))


def test_encoder_validates_instead_of_serializing_invalid_values():
    with pytest.raises(ProtocolError):
        encode_packet(replace(example_packet(), sequence=-1))
    with pytest.raises(ProtocolError):
        encode_packet(replace(example_packet(), position_m=(float("nan"), 0, 0)))


def test_namespaced_state_reference_semantics():
    state = GeomagicState((1.0, 2.0, 3.0), (0.0, 0.0, 0.6, 0.8),
                          (True, False), timestamp=7.0, valid=True)
    assert state.units == "m" and state.frame_id == "geomagic_base"
    assert state.grey_button is True and state.white_button is False
    assert state.pose_is_finite() and state.orientation_is_usable()
    assert state.server_receive_monotonic_time == 7.0
    assert not replace(state, orientation_xyzw=None).orientation_is_usable()
    assert not replace(state, orientation_xyzw=(0, 0, 0, 2)).orientation_is_usable()
