#!/usr/bin/env bash
# Run the force-disabled state streamer with a process-scoped OpenHaptics
# environment (same semantics as the validated scripts/geomagic_env.sh).
# Nothing global (.bashrc, ldconfig, udev) is touched.
#
# Usage: run_hdapi_state_stream.sh [--port 5558] [--rate-hz 500] [--duration S] ...
set -euo pipefail

device_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEOMAGIC_LOCAL="${GEOMAGIC_LOCAL:-${HOME}/.local/geomagic}"
export GTDD_HOME="${GEOMAGIC_LOCAL}/3DSystems"
export LD_LIBRARY_PATH="${GEOMAGIC_LOCAL}/lib:${GEOMAGIC_LOCAL}/compat/lib/x86_64-linux-gnu:${GEOMAGIC_LOCAL}/compat/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

binary="${device_dir}/bin/hdapi_state_stream"
if [[ ! -x "${binary}" ]]; then
  echo "run_hdapi_state_stream: binary not built: ${binary} (run build_hdapi_state_stream.sh)" >&2
  exit 1
fi
exec "${binary}" "$@"
