#!/usr/bin/env bash
# Build only; never opens the Touch. Isolated from force-off production binary.
set -euo pipefail
device_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
geo_runtime="${GEOMAGIC_LOCAL:-${HOME}/.local/geomagic}"
mkdir -p "${device_dir}/bin"
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wno-attributes -Wno-unknown-pragmas -Dlinux \
  -isystem "${geo_runtime}/include" "${device_dir}/teleop_geo_hdapi.cpp" \
  -L"${geo_runtime}/lib" -Wl,--disable-new-dtags \
  -Wl,-rpath,"${geo_runtime}/lib" \
  -Wl,-rpath,"${geo_runtime}/compat/lib/x86_64-linux-gnu" \
  -Wl,-rpath,"${geo_runtime}/compat/usr/lib/x86_64-linux-gnu" \
  -lHDU -lHD -lrt -pthread -o "${device_dir}/bin/teleop_geo_hdapi"
echo "built=${device_dir}/bin/teleop_geo_hdapi (standalone test only)"
