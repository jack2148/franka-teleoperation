"""Strict, deterministic JSON Geomagic state protocol, version 1.

All timestamps are seconds in their explicitly named clock domain. Version 1
uses geomagic_base coordinates, metre positions, and XYZW quaternions. It
preserves quaternion components (including sign) within the reference norm
tolerance of 0.05; it does not normalize or reinterpret button states.
"""

from dataclasses import asdict, dataclass
import json
import math
import re
from typing import Any, Tuple

from geomagic.vendor.geomagic_core.state import QUATERNION_NORM_TOLERANCE


PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 4096
MAX_SEQUENCE = (1 << 63) - 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", re.ASCII)
_FIELDS = frozenset({
    "protocol_version", "session_id", "sequence", "source_timestamp",
    "source_clock_domain", "position_m", "orientation_xyzw", "grey_button",
    "white_button", "valid",
})


class ProtocolError(ValueError):
    """A datagram does not satisfy the state wire contract."""


class UnsupportedVersionError(ProtocolError):
    """An explicit integer protocol version is not supported."""


@dataclass(frozen=True)
class GeomagicStatePacket:
    protocol_version: int
    session_id: str
    sequence: int
    source_timestamp: float
    source_clock_domain: str
    position_m: Tuple[float, float, float]
    orientation_xyzw: Tuple[float, float, float, float]
    grey_button: bool
    white_button: bool
    valid: bool


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ProtocolError(f"{name} must be numeric, not bool or text")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ProtocolError(f"{name} is not a finite number") from error
    if not math.isfinite(number):
        raise ProtocolError(f"{name} must be finite")
    return number


def _identifier(value: Any, name: str, limit: int) -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= limit
            or _IDENTIFIER.fullmatch(value) is None):
        raise ProtocolError(f"{name} must be a nonempty ASCII identifier (max {limit})")
    return value


def _vector(value: Any, name: str, length: int) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ProtocolError(f"{name} must have exactly {length} components")
    return tuple(_number(component, name) for component in value)


def _validate(fields: Any) -> GeomagicStatePacket:
    if not isinstance(fields, dict) or set(fields) != _FIELDS:
        raise ProtocolError("packet fields must exactly match protocol version 1")
    version = fields["protocol_version"]
    if type(version) is not int:
        raise ProtocolError("protocol_version must be an integer")
    if version != PROTOCOL_VERSION:
        raise UnsupportedVersionError(f"unsupported protocol version: {version}")
    sequence = fields["sequence"]
    if type(sequence) is not int or not 0 <= sequence <= MAX_SEQUENCE:
        raise ProtocolError("sequence must be an integer in [0, 2**63-1]")
    timestamp = _number(fields["source_timestamp"], "source_timestamp")
    if timestamp < 0:
        raise ProtocolError("source_timestamp must be nonnegative seconds")
    for name in ("grey_button", "white_button", "valid"):
        if type(fields[name]) is not bool:
            raise ProtocolError(f"{name} must be a JSON boolean")
    position = _vector(fields["position_m"], "position_m", 3)
    orientation = _vector(fields["orientation_xyzw"], "orientation_xyzw", 4)
    norm = math.hypot(*orientation)
    if not math.isfinite(norm) or abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ProtocolError("quaternion norm must be within 0.05 of 1.0")
    return GeomagicStatePacket(
        protocol_version=version,
        session_id=_identifier(fields["session_id"], "session_id", 128),
        sequence=sequence,
        source_timestamp=timestamp,
        source_clock_domain=_identifier(fields["source_clock_domain"], "source_clock_domain", 64),
        position_m=position,
        orientation_xyzw=orientation,
        grey_button=fields["grey_button"],
        white_button=fields["white_button"],
        valid=fields["valid"],
    )


def encode_packet(packet: GeomagicStatePacket) -> bytes:
    """Validate, then emit canonical compact UTF-8 JSON with sorted keys."""
    if not isinstance(packet, GeomagicStatePacket):
        raise ProtocolError("encode_packet expects GeomagicStatePacket")
    validated = _validate(asdict(packet))
    payload = json.dumps(asdict(validated), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    if len(payload) > MAX_PACKET_BYTES:
        raise ProtocolError("packet exceeds size limit")
    return payload


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON constant: {value}")


def decode_packet(payload: bytes) -> GeomagicStatePacket:
    """Parse untrusted bytes without pickle, evaluation, or code execution."""
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_PACKET_BYTES:
        raise ProtocolError("packet must be 1..4096 bytes")
    try:
        fields = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except ProtocolError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ProtocolError("malformed UTF-8 JSON packet") from error
    return _validate(fields)
