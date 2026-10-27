// Hardware-free tests of the exact policy/cleanup used by the HDAPI owner.
#include "../device/teleop_geo_force.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace teleop_geo;

Vec3 steady(ForceGate& gate, const Vec3& vector) {
  Vec3 out{};
  for (int i = 1; i <= 1000; ++i) {
    const double time = 1 + i * .001;
    out = gate.step({vector, time, static_cast<std::uint64_t>(i), true},
                    time, .001, true, false, true, true, true, false);
    assert(norm(out) <= kHardCapN + 1e-14);
  }
  return out;
}

struct FakeDevice {
  double force{1};
  std::vector<std::string> events;
  void zero_synchronously() { events.push_back("zero_sync"); force = 0; }
  void stop_callback_and_scheduler() { events.push_back("stop"); assert(force == 0); }
  void disable_force() { events.push_back("disable"); assert(force == 0); }
  void close_device() { events.push_back("close"); assert(force == 0); }
};

int main() {
  int tests = 0;
  for (const Vec3 vector : {Vec3{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}, {100,100,100}}) {
    ForceGate gate;
    const auto out = steady(gate, vector);
    assert(std::abs(norm(out) - 1) < 1e-14);
    for (int axis = 0; axis < 3; ++axis)
      assert(std::abs(out[axis] - vector[axis] / norm(vector)) < 1e-14);
    ++tests;
  }
  {
    ForceGate gate;
    const auto first = gate.step({{1,0,0}, 1, 1, true}, 1, .001, true, false, true, true, true, false);
    assert(std::abs(norm(first) - .003) < 1e-15);  // 3 N/s onset, not a 1 N step.
    ++tests;
  }
  {
    ForceGate gate;
    steady(gate, {1,0,0});
    const Command cmd{{1,0,0}, 2, 1001, true};
    assert(norm(gate.step(cmd, 2.019, .001, true, false, true, true, true, false)) == 1);
    assert(norm(gate.step(cmd, 2.021, .001, true, false, true, true, true, false)) == 0);
    assert(gate.stale);
    assert(norm(gate.step(cmd, 2.022, .001, true, false, true, true, true, false)) == 0);
    ++tests;
  }
  for (int reason = 0; reason < 11; ++reason) {
    ForceGate gate;
    steady(gate, {1,0,0});
    Command cmd{{1,0,0}, 2, 1001, true};
    if (reason == 7) cmd.force = {};
    if (reason == 8) cmd.force[0] = std::numeric_limits<double>::quiet_NaN();
    if (reason == 9) cmd.timestamp = 3;
    const auto zero = gate.step(cmd, 2, reason == 10 ? .1 : .001,
      reason != 0, reason == 1, reason != 2, reason != 3, reason != 4, reason == 5, reason != 6);
    assert(norm(zero) == 0);
    // Replaying a pre-release packet cannot re-arm even after grey/state recovers.
    assert(norm(gate.step({{1,0,0}, 2, 1001, true}, 2, .001, true, false, true, true, true, false)) == 0);
    assert(norm(gate.step({{1,0,0}, 2, 1002, true}, 2, .001, true, false, true, true, true, false)) > 0);
    ++tests;
  }
  {
    Packet packet{kMagic, 42, 5, 2, {1,0,0}, 1, 0};
    assert(valid_packet(packet, 42, 4, 2));
    assert(!valid_packet(packet, 43, 4, 2));
    assert(!valid_packet(packet, 42, 5, 2));
    assert(!valid_packet(packet, 42, 4, 2.021));
    assert(!valid_packet(packet, 42, 4, 1.99));
    packet.reserved = 1; assert(!valid_packet(packet, 42, 4, 2));
    packet.reserved = 0; packet.enabled = 2; assert(!valid_packet(packet, 42, 4, 2));
    packet.enabled = 1; packet.force[1] = std::numeric_limits<double>::infinity();
    assert(!valid_packet(packet, 42, 4, 2));
    ++tests;
  }
  for (bool exceptional : {false, true}) {
    FakeDevice device;
    try {
      if (exceptional) throw std::runtime_error("simulated controller exception");
    } catch (const std::exception&) {}
    zero_then_close(device);
    assert((device.events == std::vector<std::string>{"zero_sync","disable","stop","disable","close"}));
    assert(device.force == 0);
    ++tests;
  }
  std::cout << "teleop_geo native force tests: " << tests << " passed, 0 failed, 0 skipped\n";
}
