#!/usr/bin/env bash
# Build the force-disabled HDAPI state streamer against the server-local
# OpenHaptics runtime. Flags follow the validated reference
# geomagic_franka_teleop_import_v2/scripts/build_haptic_servo.sh.
set -euo pipefail

device_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEOMAGIC_LOCAL="${GEOMAGIC_LOCAL:-${HOME}/.local/geomagic}"
output="${device_dir}/bin/hdapi_state_stream"
mkdir -p "${device_dir}/bin"

# DT_RPATH (not RUNPATH) so libHD's own dependencies (libPhantomIOLib42,
# libncurses.so.5, libtinfo.so.5) resolve from the server-local runtime too.
# libHDU is static (libHDU.a) and references libHD, so it precedes -lHD.
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -Wno-attributes -Wno-unknown-pragmas -Dlinux \
  -I"${GEOMAGIC_LOCAL}/include" \
  "${device_dir}/hdapi_state_stream.cpp" \
  -L"${GEOMAGIC_LOCAL}/lib" \
  -Wl,--disable-new-dtags \
  -Wl,-rpath,"${GEOMAGIC_LOCAL}/lib" \
  -Wl,-rpath,"${GEOMAGIC_LOCAL}/compat/lib/x86_64-linux-gnu" \
  -Wl,-rpath,"${GEOMAGIC_LOCAL}/compat/usr/lib/x86_64-linux-gnu" \
  -lHDU -lHD -lrt -pthread \
  -o "${output}"

echo "built=${output}"
