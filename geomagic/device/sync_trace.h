// Opt-in bounded diagnostic storage. One callback writer and one publisher
// writer own separate preallocated arrays. Pause capture before serialization;
// the device callback continues normally while capture is paused.
// No files, locks, allocation, JSON, or socket traffic in the servo callback.
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace geo_sync {
inline std::uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline std::uint64_t ns(double seconds) { return std::llround(seconds * 1e9); }
struct ServoRecord {
  std::uint64_t callback{}, acquired{}, received{}, applied{}, committed{}, command{};
  std::uint64_t sample{}, force_sequence{};
  std::array<double,3> position{}, force{}, requested{};
  std::array<double,4> quaternion{};
  int buttons{};
  bool stale{}, valid{};
};
struct PacketRecord {
  std::uint64_t stamp{}, sequence{}, sample{}, callback{};
};
class Trace {
  // Lock-free handshake with the single servo writer. The publisher owns
  // packet storage and calls begin/checkpoint outside the callback.
  std::atomic<bool> servo_enabled{true}, servo_busy{false};
  bool packet_enabled{true}, saved{false};
 public:
  std::string directory, session;
  std::vector<ServoRecord> servo_records;
  std::vector<PacketRecord> packet_records;
  std::size_t servo_count{}, packet_count{};
  std::uint64_t servo_start{}, packet_start{}, duration{};
  bool servo_limited{}, packet_limited{};
  Trace(const std::string& path, double seconds, std::size_t capacity)
      : directory(path), servo_records(capacity), packet_records(capacity),
        duration(ns(seconds)) {
    if (!std::isfinite(seconds) || seconds <= 0 || seconds > 600 || capacity < 1 || capacity > 500000)
      throw std::runtime_error("invalid bounded synchronization capture options");
  }
  static std::unique_ptr<Trace> from_environment() {
    const char* path = std::getenv("GEOMAGIC_SYNC_CAPTURE");
    if (!path || !*path) return nullptr;
    const char* seconds = std::getenv("GEOMAGIC_SYNC_SECONDS");
    const char* records = std::getenv("GEOMAGIC_SYNC_RECORDS");
    double limit = seconds ? std::stod(seconds) : 60.;
    long capacity = records ? std::stol(records) : 200000;
    // Validate before allocating or opening a physical device.
    if (!std::isfinite(limit) || limit <= 0 || limit > 600 || capacity < 1 || capacity > 500000)
      throw std::runtime_error("invalid bounded synchronization capture options");
    return std::make_unique<Trace>(path, limit, capacity);
  }
  void servo(const ServoRecord& record) noexcept {
    if (!servo_enabled.load()) return;
    servo_busy.store(true);
    if (!servo_enabled.load()) { servo_busy.store(false); return; }
    if (!servo_start) servo_start = record.callback;
    if (servo_count == servo_records.size() || record.callback-servo_start > duration) {
      servo_limited = true;
      servo_busy.store(false);
      return;
    }
    servo_records[servo_count++] = record;
    servo_busy.store(false);
  }
  void packet(const PacketRecord& record) noexcept {
    if (!packet_enabled) return;
    if (!packet_start) packet_start = record.stamp;
    if (packet_count == packet_records.size() || record.stamp-packet_start > duration) {
      packet_limited = true;
      return;
    }
    packet_records[packet_count++] = record;
  }
  void pause() noexcept {
    servo_enabled.store(false);
    packet_enabled = false;
    while (servo_busy.load()) std::this_thread::yield();
  }
  void wait_for_begin() noexcept { pause(); saved = true; }
  void begin(const std::string& path) {
    pause();
    if (!saved && (servo_count || packet_count))
      throw std::runtime_error("previous capture has not been saved");
    if (path.empty() || path.front() != '/' || path.find_first_of("\t\r\n") != std::string::npos)
      throw std::runtime_error("capture directory must be an absolute path without control characters");
    if (std::filesystem::exists(path + "/native-meta-" + std::to_string(getpid()) + ".json"))
      throw std::runtime_error("capture directory already contains this native session");
    directory = path;
    servo_count = packet_count = servo_start = packet_start = 0;
    servo_limited = packet_limited = saved = false;
    packet_enabled = true;
    servo_enabled.store(true);
  }
  void checkpoint() {
    pause();
    if (saved) return;
    write();
    saved = true;
  }
  void write() const {
    std::filesystem::create_directories(directory);
    const std::string suffix = "-" + std::to_string(getpid());
    const std::string servo_path = directory + "/native-servo" + suffix + ".csv";
    const std::string packet_path = directory + "/native-packets" + suffix + ".csv";
    const std::string meta_path = directory + "/native-meta" + suffix + ".json";
    std::ofstream out(servo_path + ".tmp");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.precision(17);
    out << "monotonic_ns,sim_time,session,sequence,sample_index,callback_ns,acquired_ns,H3_ns,H4_ns,committed_ns,command_ns,buttons,stale,valid,fx,fy,fz,cfx,cfy,cfz,px,py,pz,qx,qy,qz,qw\n";
    for (std::size_t i=0; i<servo_count; ++i) {
      const auto& r=servo_records[i];
      out << (r.applied ? r.applied : r.acquired) << ",," << session << ',' << r.force_sequence
          << ',' << r.sample << ',' << r.callback << ',' << r.acquired << ',' << r.received
          << ',' << r.applied << ',' << r.committed << ',' << r.command << ',' << r.buttons
          << ',' << r.stale << ',' << r.valid;
      for (const auto& v : {r.force, r.requested, r.position}) for (double x : v) out << ',' << x;
      for (double x : r.quaternion) out << ',' << x;
      out << '\n';
    }
    out.close();
    std::ofstream packets(packet_path + ".tmp");
    packets.exceptions(std::ios::failbit | std::ios::badbit);
    packets << "monotonic_ns,sim_time,session,sequence,sample_index,callback_ns\n";
    for (std::size_t i=0; i<packet_count; ++i) {
      const auto& r=packet_records[i];
      packets << r.stamp << ",," << session << ',' << r.sequence << ',' << r.sample << ',' << r.callback << '\n';
    }
    packets.close();
    std::ofstream meta(meta_path + ".tmp");
    meta.exceptions(std::ios::failbit | std::ios::badbit);
    meta << "{\"clock\":\"linux.CLOCK_MONOTONIC\",\"monotonic_ns\":" << now_ns()
         << ",\"sim_time\":null,\"session\":\"" << session << "\",\"sequence\":null,"
         << "\"seconds_limit\":" << duration / 1e9 << ",\"record_limit\":" << servo_records.size() << ','
         << "\"servo_records\":" << servo_count << ",\"packet_records\":" << packet_count
         << ",\"servo_limited\":" << (servo_limited ? "true":"false")
         << ",\"packet_limited\":" << (packet_limited ? "true":"false")
         << ",\"sim_time_note\":\"joined from host force/state sequence; hardware does not own simulation time\"}\n";
    meta.close();
    std::filesystem::rename(servo_path + ".tmp", servo_path);
    std::filesystem::rename(packet_path + ".tmp", packet_path);
    // Metadata appears last, marking a complete native capture.
    std::filesystem::rename(meta_path + ".tmp", meta_path);
  }
};
static_assert(std::atomic<bool>::is_always_lock_free, "capture handshake must be lock-free");
}  // namespace geo_sync
