#pragma once
// Isolated test-servo preprocessing. Cap, vector slew, bounded snapshots and
// immediate safety-zero semantics follow import_v2/hdapi_haptic_servo.cpp.
// No SDK/device calls: this exact policy is exercised by the offline C++ tests.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace teleop_geo {
using Vec3 = std::array<double, 3>;
constexpr double kHardCapN = 1.0;
constexpr double kTimeoutS = .020;
constexpr double kSlewNPerS = 3.0;
constexpr std::uint64_t kMagic = 0x314543524f464754ULL;  // little-endian TGFORCE1
struct Packet {
  std::uint64_t magic, token, sequence;
  double timestamp;
  Vec3 force;
  std::uint32_t enabled, reserved;
};
static_assert(sizeof(Packet) == 64, "force IPC ABI must match Python <8sQQd3dII");
static_assert(sizeof(Vec3) == 24, "force vector ABI");

inline double norm(const Vec3& v) { return std::hypot(v[0], v[1], v[2]); }
inline bool finite(const Vec3& v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}
inline bool valid_packet(const Packet& p, std::uint64_t token,
                         std::uint64_t previous_sequence, double now) {
  return p.magic == kMagic && p.token == token && p.sequence > previous_sequence &&
      p.enabled <= 1 && p.reserved == 0 && finite(p.force) && std::isfinite(p.timestamp) &&
      std::isfinite(now) && p.timestamp <= now + .001 && now - p.timestamp <= kTimeoutS;
}

struct Command {
  Vec3 force{};
  double timestamp{};
  std::uint64_t sequence{};
  bool enabled{};
};

class ForceGate {
 public:
  Vec3 previous{};
  std::uint64_t barrier{};
  bool stale{true};
  bool active{};

  Vec3 step(const Command& cmd, double now, double dt, bool grey, bool white,
            bool state_valid, bool calibration_valid, bool output_enabled,
            bool stopping, bool coherent = true) {
    stale = !std::isfinite(now) || !std::isfinite(cmd.timestamp) ||
        cmd.timestamp > now + .001 || now - cmd.timestamp >= kTimeoutS;
    const double magnitude = norm(cmd.force);
    active = coherent && !stale && grey && !white && state_valid && calibration_valid &&
        output_enabled && !stopping && cmd.enabled && cmd.sequence > barrier &&
        finite(cmd.force) && std::isfinite(magnitude) && magnitude > 1e-12 &&
        std::isfinite(dt) && dt > 0 && dt <= kTimeoutS;
    if (!active) {
      previous = {};
      barrier = std::max(barrier, cmd.sequence);
      return previous;  // Release, stale, invalid and explicit zero bypass slew.
    }
    Vec3 target = cmd.force;
    if (magnitude > kHardCapN)
      for (double& value : target) value *= kHardCapN / magnitude;
    Vec3 delta{};
    for (int i = 0; i < 3; ++i) delta[i] = target[i] - previous[i];
    const double distance = norm(delta);
    const double amount = std::min(1.0, kSlewNPerS * dt / std::max(distance, 1e-300));
    for (int i = 0; i < 3; ++i) previous[i] += amount * delta[i];
    // Convex-segment slew cannot widen the cap; guard floating-point roundoff.
    const double out_norm = norm(previous);
    if (out_norm > kHardCapN)
      for (double& value : previous) value *= kHardCapN / out_norm;
    return previous;
  }
};

// Used by both the real HDAPI owner and the fake device lifecycle test.
// The sole active servo callback acknowledges zero before output is disabled.
// Disable also covers a fault/stalled scheduler that cannot acknowledge zero.
// No extra begin/end frame is issued by cleanup, even on exceptional unwinding.
template <typename Device>
void zero_then_close(Device& device) noexcept {
  device.zero_synchronously();
  device.disable_force();
  device.stop_callback_and_scheduler();
  device.disable_force();  // Also after join: no late callback can re-enable it.
  device.close_device();
}
}  // namespace teleop_geo
