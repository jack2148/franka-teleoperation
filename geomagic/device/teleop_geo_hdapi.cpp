// Dedicated standalone test owner. Production hdapi_state_stream stays OFF.
// Acquisition/calibration, quaternion extraction, atomic latest-state pattern
// derive from the validated import_v2 reference and force-off streamer.
// Force gates/slew derive from import_v2's direct haptic servo. No MuJoCo,
// JSON, sockets, files or printing in the scheduler callback.
#include "teleop_geo_force.h"
#include "sync_trace.h"
#include <HD/hd.h>
#include <HDU/hduError.h>
#include <HDU/hduMatrix.h>
#include <HDU/hduQuaternion.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "teleop_geo force IPC requires a little-endian host"
#endif
using namespace teleop_geo;
using Clock = std::chrono::steady_clock;
static_assert(std::atomic<double>::is_always_lock_free, "lock-free servo required");
static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "lock-free servo required");
namespace {
std::atomic<bool> stopping{false};
void signal_handler(int) { stopping.store(true); }
double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
std::string json_string(const char* input) {
  std::string out = "\"";
  for (const char* p = input ? input : "UNAVAILABLE"; *p; ++p) {
    if (*p == '"' || *p == '\\') out += '\\';
    if (static_cast<unsigned char>(*p) >= 32) out += *p;
  }
  return out + '"';
}

struct Shared {
  geo_sync::Trace* sync{};
  std::atomic<std::uint64_t> force_received_ns{0};
  std::atomic<std::uint64_t> state_version{0}, sample{0}, callbacks{0};
  std::array<std::atomic<double>, 3> position{}, actual_force{};
  std::array<std::atomic<double>, 4> quaternion{};
  std::atomic<double> timestamp{0};
  std::atomic<int> buttons{0};
  std::atomic<bool> fault{false}, stale{true}, force_active{false};
  std::atomic<int> error_code{0}, internal_error_code{0};
  std::atomic<bool> ready{false}, stop_zero_written{false};
  std::atomic<std::uint64_t> force_version{0}, force_sequence{0};
  std::array<std::atomic<double>, 3> force{};
  std::atomic<double> force_time{0};
  std::atomic<bool> force_enabled{false};
  ForceGate gate;
  double previous_tick{};
  bool output_initialized{};  // Scheduler thread only.
  pid_t parent{};

  void publish(const Command& c, std::uint64_t received_ns=0) {
    force_version.fetch_add(1);
    for (int i = 0; i < 3; ++i) force[i].store(c.force[i]);
    force_time.store(c.timestamp);
    force_sequence.store(c.sequence);
    force_enabled.store(c.enabled);
    if (sync) force_received_ns.store(received_ns);
    force_version.fetch_add(1);
  }
  bool command(Command& c, std::uint64_t* received_ns=nullptr) const {
    const auto before = force_version.load();
    if (before & 1) return false;
    for (int i = 0; i < 3; ++i) c.force[i] = force[i].load();
    c.timestamp = force_time.load();
    c.sequence = force_sequence.load();
    c.enabled = force_enabled.load();
    if (received_ns) *received_ns = force_received_ns.load();
    return before == force_version.load();  // One bounded attempt, fail to zero.
  }
};

HDCallbackCode HDCALLBACK servo_callback(void* argument) {
  auto& s = *static_cast<Shared*>(argument);
  const double now = now_s();
  const double dt = s.previous_tick ? now - s.previous_tick : .001;
  s.previous_tick = now;
  s.callbacks.fetch_add(1);
  const bool calibration = hdCheckCalibration() == HD_CALIBRATION_OK;
  HDdouble position[3]{}, matrix[16]{};
  HDint buttons = 0;
  hdBeginFrame(hdGetCurrentDevice());
  const bool initializing = !s.output_initialized;
  if (initializing && !stopping.load() && getppid() == s.parent) hdEnable(HD_FORCE_OUTPUT);
  hdGetDoublev(HD_CURRENT_POSITION, position);
  hdGetDoublev(HD_CURRENT_TRANSFORM, matrix);
  hdGetIntegerv(HD_CURRENT_BUTTONS, &buttons);
  const auto acquired_ns = s.sync ? geo_sync::now_ns() : 0;
  HDErrorInfo error = hdGetError();
  bool valid = !HD_DEVICE_ERROR(error) && calibration;
  for (double value : position) valid = valid && std::isfinite(value);
  for (double value : matrix) valid = valid && std::isfinite(value);
  Vec3 p{};
  std::array<double, 4> q{};
  if (valid) {
    hduMatrix transform(matrix), rotation;
    transform.getRotationMatrix(rotation);
    const hduQuaternion quat(rotation);
    q = {quat.v()[0], quat.v()[1], quat.v()[2], quat.s()};
    const double normq = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    valid = std::isfinite(normq) && std::abs(normq - 1) <= .05;
    if (valid) {
      for (double& value : q) value /= normq;
      if (q[3] < 0) for (double& value : q) value = -value;
      for (int i = 0; i < 3; ++i) p[i] = position[i] * .001;
    }
  }
  Command command;
  std::uint64_t received_ns=0;
  const bool coherent = s.command(command, s.sync ? &received_ns : nullptr);
  const bool stop = stopping.load() || getppid() != s.parent;
  if (stop) stopping.store(true);
  const Vec3 force = s.gate.step(command, now, dt,
      (buttons & HD_DEVICE_BUTTON_1) != 0, (buttons & HD_DEVICE_BUTTON_2) != 0,
      valid, calibration, hdIsEnabled(HD_FORCE_OUTPUT) == HD_TRUE, stop || initializing, coherent);
  hdSetDoublev(HD_CURRENT_FORCE, force.data());
  const auto applied_ns = s.sync ? geo_sync::now_ns() : 0;
  hdEndFrame(hdGetCurrentDevice());
  const auto committed_ns = s.sync ? geo_sync::now_ns() : 0;
  const HDErrorInfo end_error = hdGetError();
  if (s.sync) s.sync->servo({geo_sync::ns(now), acquired_ns, coherent ? received_ns : 0,
      applied_ns, committed_ns, geo_sync::ns(command.timestamp),
      s.sample.load() + (valid ? 1 : 0), command.sequence, p, force, command.force, q,
      buttons, s.gate.stale, valid && !HD_DEVICE_ERROR(error) && !HD_DEVICE_ERROR(end_error)});
  if (HD_DEVICE_ERROR(error) || HD_DEVICE_ERROR(end_error)) {
    const auto fault = HD_DEVICE_ERROR(error) ? error : end_error;
    s.error_code.store(static_cast<int>(fault.errorCode));
    s.internal_error_code.store(fault.internalErrorCode);
    // The frame has ended: opening a second one in this tick is illegal.
    // A read fault already commanded zero; any end-frame fault disables
    // hardware output immediately instead of trying another force frame.
    hdDisable(HD_FORCE_OUTPUT);
    for (auto& value : s.actual_force) value.store(0);
    s.fault.store(true);
    stopping.store(true);
    return HD_CALLBACK_DONE;
  }
  s.output_initialized = true;
  if (stop) s.stop_zero_written.store(true);  // Ack only after successful zero EndFrame.
  s.state_version.fetch_add(1);
  if (valid && !stop) s.ready.store(true);  // First valid state and successful zero startup frame.
  if (valid) {
    for (int i = 0; i < 3; ++i) s.position[i].store(p[i]);
    for (int i = 0; i < 4; ++i) s.quaternion[i].store(q[i]);
    s.buttons.store(buttons);
    s.timestamp.store(now);
    s.sample.fetch_add(1);
  }
  for (int i = 0; i < 3; ++i) s.actual_force[i].store(force[i]);
  s.stale.store(s.gate.stale);
  s.force_active.store(s.gate.active);
  s.state_version.fetch_add(1);
  return stop ? HD_CALLBACK_DONE : HD_CALLBACK_CONTINUE;
}

struct Device {
  HHD handle{HD_INVALID_HANDLE};
  HDSchedulerHandle callback{HD_INVALID_HANDLE};
  bool scheduler{};
  bool closed{};
  Shared* shared{};
  void zero_synchronously() noexcept {
    if (handle == HD_INVALID_HANDLE || !scheduler || callback == HD_INVALID_HANDLE || !shared) return;
    stopping.store(true);
    // The existing callback owns the ONLY frame. Do not schedule a competing
    // zero-frame callback. Wait outside the scheduler for its stop-zero ack.
    const double deadline = now_s() + .050;
    while (!shared->stop_zero_written.load() && !shared->fault.load() && now_s() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  void stop_callback_and_scheduler() noexcept {
    if (callback != HD_INVALID_HANDLE) hdUnschedule(callback);
    if (scheduler) hdStopScheduler();
    scheduler = false;
  }
  void disable_force() noexcept { if (handle != HD_INVALID_HANDLE) hdDisable(HD_FORCE_OUTPUT); }
  void close_device() noexcept {
    if (handle != HD_INVALID_HANDLE) hdDisableDevice(handle);
    handle = HD_INVALID_HANDLE;
    closed = true;
  }
  ~Device() { if (!closed) { stopping.store(true); zero_then_close(*this); } }
};

bool calibrate() {
  HDint styles = 0;
  hdGetIntegerv(HD_CALIBRATION_STYLE, &styles);
  const HDenum style = styles & HD_CALIBRATION_INKWELL ? HD_CALIBRATION_INKWELL :
      styles & HD_CALIBRATION_ENCODER_RESET ? HD_CALIBRATION_ENCODER_RESET : HD_CALIBRATION_AUTO;
  const double deadline = now_s() + 30;
  std::cout << "{\"event\":\"calibration\",\"instruction\":\"Place stylus fully in inkwell if requested\"}" << std::endl;
  while (!stopping.load() && now_s() < deadline) {
    const HDenum state = hdCheckCalibration();
    if (state == HD_CALIBRATION_OK) return true;
    if (state == HD_CALIBRATION_NEEDS_UPDATE) {
      hdUpdateCalibration(style);
      if (HD_DEVICE_ERROR(hdGetError())) return false;
    } else if (state != HD_CALIBRATION_NEEDS_MANUAL_INPUT) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return false;
}

void receive_force(int fd, std::uint64_t token, Shared& s) {
  std::uint64_t sequence = 0;
  while (!stopping.load()) {
    pollfd descriptor{fd, POLLIN, 0};
    const int result = poll(&descriptor, 1, 5);
    if (result < 0 && errno != EINTR) { stopping.store(true); break; }
    if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) { stopping.store(true); break; }
    if (!(descriptor.revents & POLLIN)) continue;
    for (int budget = 0; budget < 64; ++budget) {
      Packet packet{};
      const auto length = recv(fd, &packet, sizeof(packet), MSG_DONTWAIT | MSG_TRUNC);
      if (length < 0) break;
      if (length == sizeof(packet) && valid_packet(packet, token, sequence, now_s())) {
        sequence = packet.sequence;
        s.publish({packet.force, packet.timestamp, packet.sequence, packet.enabled == 1},
                  s.sync ? geo_sync::now_ns() : 0);
      } else {
        s.publish({{}, 0, sequence, false});  // Never refresh invalid/old commands.
      }
    }
  }
  s.publish({{}, 0, sequence, false});
}

struct Snapshot {
  std::uint64_t sample{};
  double timestamp{};
  Vec3 p{}, force{};
  std::array<double, 4> q{};
  int buttons{};
  bool stale{}, active{};
};
bool snapshot(const Shared& s, Snapshot& out) {
  const auto before = s.state_version.load();
  if (before & 1) return false;
  out.sample = s.sample.load();
  out.timestamp = s.timestamp.load();
  out.buttons = s.buttons.load();
  out.stale = s.stale.load();
  out.active = s.force_active.load();
  for (int i = 0; i < 3; ++i) { out.p[i] = s.position[i].load(); out.force[i] = s.actual_force[i].load(); }
  for (int i = 0; i < 4; ++i) out.q[i] = s.quaternion[i].load();
  return before == s.state_version.load();
}
}  // namespace

int main(int argc, char** argv) {
  int fd = -1, port = 0, sync_control_fd = -1;
  std::uint64_t token = 0;
  pid_t parent = 0;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string key(argv[i]);
      if (i + 1 >= argc) throw std::runtime_error("missing option value");
      const std::string value(argv[++i]);
      if (key == "--force-fd") fd = std::stoi(value);
      else if (key == "--sync-control-fd") sync_control_fd = std::stoi(value);
      else if (key == "--state-port") port = std::stoi(value);
      else if (key == "--token") token = std::stoull(value);
      else if (key == "--parent-pid") parent = std::stoi(value);
      else throw std::runtime_error("unknown option");
    }
    if (fd < 3 || port < 1 || port > 65535 || !token || parent != getppid())
      throw std::runtime_error("requires private parent-owned force channel");
    int socket_type = 0, socket_domain = 0; socklen_t size = sizeof(socket_type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &socket_type, &size) || socket_type != SOCK_DGRAM)
      throw std::runtime_error("force channel must be a datagram socketpair");
    size = sizeof(socket_domain);
    if (getsockopt(fd, SOL_SOCKET, SO_DOMAIN, &socket_domain, &size) || socket_domain != AF_UNIX)
      throw std::runtime_error("force channel must be private AF_UNIX");
    ucred peer{}; size = sizeof(peer);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) ||
        peer.pid != parent || peer.uid != getuid())
      throw std::runtime_error("force channel peer must be the launch parent");
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || parent != getppid())
      throw std::runtime_error("parent-death guard failed");
  } catch (const std::exception& e) {
    std::cerr << "teleop_geo_hdapi configuration: " << e.what() << '\n';
    return 64;  // No device initialization on malformed invocation.
  }
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  Shared state;
  std::unique_ptr<geo_sync::Trace> sync;
  try {
    sync = geo_sync::Trace::from_environment();
    if (sync) {
      sync->session = "teleop-geo-" + std::to_string(token);
      state.sync = sync.get();
      if (sync_control_fd >= 0) sync->wait_for_begin();
    } else if (sync_control_fd >= 0) {
      throw std::runtime_error("sync-control-fd requires GEOMAGIC_SYNC_CAPTURE");
    }
  } catch (const std::exception& e) {
    std::cerr << "synchronization configuration: " << e.what() << '\n';
    return 64;
  }
  Device device;
  device.shared = &state;
  state.parent = parent;
  std::thread receiver;
  int state_socket = -1;
  int code = 0;
  try {
    state_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (state_socket < 0) throw std::runtime_error("state socket failed");
    device.handle = hdInitDevice(HD_DEFAULT_DEVICE);  // Exactly one device owner.
    if (HD_DEVICE_ERROR(hdGetError())) throw std::runtime_error("hdInitDevice failed");
    hdDisable(HD_FORCE_OUTPUT);
    std::cout << "{\"event\":\"device\",\"model\":" << json_string(hdGetString(HD_DEVICE_MODEL_TYPE))
              << ",\"serial\":" << json_string(hdGetString(HD_DEVICE_SERIAL_NUMBER)) << "}" << std::endl;
    hdStartScheduler();
    if (HD_DEVICE_ERROR(hdGetError())) throw std::runtime_error("scheduler failed");
    device.scheduler = true;
    if (!calibrate()) throw std::runtime_error("calibration not completed");
    receiver = std::thread(receive_force, fd, token, std::ref(state));
    device.callback = hdScheduleAsynchronous(servo_callback, &state, HD_MAX_SCHEDULER_PRIORITY);
    if (HD_DEVICE_ERROR(hdGetError())) throw std::runtime_error("callback scheduling failed");
    const double ready_deadline = now_s() + 5;
    while (!state.ready.load() && !state.fault.load() && !stopping.load() && now_s() < ready_deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!state.ready.load() && !state.fault.load() && !stopping.load())
      throw std::runtime_error("first valid HDAPI servo frame timed out");
    if (state.ready.load() && !state.fault.load() && !stopping.load())
      std::cout << "{\"event\":\"ready\",\"force_cap_n\":1.0,\"watchdog_ms\":20}" << std::endl;
    sockaddr_in destination{};
    destination.sin_family = AF_INET;
    destination.sin_port = htons(static_cast<std::uint16_t>(port));
    destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const std::string session = "teleop-geo-" + std::to_string(token);
    std::uint64_t sequence = 0, last_sample = 0, published = 0;
    std::uint64_t previous_published = 0, previous_callbacks = 0;
    auto next = Clock::now();
    double status_time = now_s();
    char buffer[768];
    Snapshot sample;
    while (!stopping.load() && !state.fault.load() && getppid() == parent) {
      // Optional UI capture control belongs to the publisher, never the servo.
      if (sync_control_fd >= 0) {
        char request[4096];
        const auto length = recv(sync_control_fd, request, sizeof(request), MSG_DONTWAIT);
        if (length > 0) {
          std::string reply;
          try {
            const std::string command(request, static_cast<std::size_t>(length));
            if (command.rfind("BEGIN\t", 0) == 0) sync->begin(command.substr(6));
            else if (command == "SAVE") sync->checkpoint();
            else throw std::runtime_error("unknown capture command");
            reply = "OK\t" + sync->directory;
          } catch (const std::exception& e) { reply = "ERROR\t" + std::string(e.what()); }
          send(sync_control_fd, reply.data(), reply.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
        }
      }
      next += std::chrono::milliseconds(2);
      std::this_thread::sleep_until(next);
      if (Clock::now() - next > std::chrono::milliseconds(10)) next = Clock::now();
      const double now = now_s();
      if (snapshot(state, sample) && sample.sample > last_sample && now - sample.timestamp < .010) {
        last_sample = sample.sample;
        const int length = std::snprintf(buffer, sizeof(buffer),
            "{\"protocol_version\":1,\"session_id\":\"%s\",\"sequence\":%llu,"
            "\"source_timestamp\":%.9f,\"source_clock_domain\":\"server.monotonic.hdapi\","
            "\"position_m\":[%.9f,%.9f,%.9f],\"orientation_xyzw\":[%.12f,%.12f,%.12f,%.12f],"
            "\"grey_button\":%s,\"white_button\":%s,\"valid\":true}",
            session.c_str(), static_cast<unsigned long long>(sequence), sample.timestamp,
            sample.p[0], sample.p[1], sample.p[2], sample.q[0], sample.q[1], sample.q[2], sample.q[3],
            sample.buttons & HD_DEVICE_BUTTON_1 ? "true" : "false",
            sample.buttons & HD_DEVICE_BUTTON_2 ? "true" : "false");
        if (length > 0 && length < static_cast<int>(sizeof(buffer)) &&
            sendto(state_socket, buffer, length, MSG_DONTWAIT,
                   reinterpret_cast<sockaddr*>(&destination), sizeof(destination)) == length) {
          ++sequence;
          ++published;
          if (sync) sync->packet({geo_sync::now_ns(), sequence-1, sample.sample, geo_sync::ns(sample.timestamp)});
        }
      }
      if (now - status_time >= 1) {
        const double period = now - status_time;
        const auto callbacks = state.callbacks.load();
        std::cout << "{\"event\":\"hdapi_status\",\"servo_hz\":" << (callbacks - previous_callbacks) / period
                  << ",\"state_hz\":" << (published - previous_published) / period
                  << ",\"rendered_force_n\":" << norm(sample.force)
                  << ",\"force_device\":[" << sample.force[0] << ',' << sample.force[1] << ',' << sample.force[2]
                  << "],\"watchdog\":\"" << (sample.stale ? "STALE" : sample.active ? "ACTIVE" : "ZERO")
                  << "\"}" << std::endl;
        previous_callbacks = callbacks;
        previous_published = published;
        status_time = now;
      }
    }
    if (state.fault.load()) {
      code = 9;
      // Main-thread diagnostic only: no allocation or printing in the servo.
      std::cout << "{\"event\":\"error\",\"code\":" << state.error_code.load()
                << ",\"internal_code\":" << state.internal_error_code.load()
                << ",\"description\":" << json_string(hdGetErrorString(state.error_code.load()))
                << "}" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "teleop_geo_hdapi: " << e.what() << '\n';
    code = 2;
  }
  stopping.store(true);
  if (receiver.joinable()) receiver.join();
  zero_then_close(device);
  if (sync) {
    try { sync->write(); }
    catch (const std::exception& e) { std::cerr << "synchronization report: " << e.what() << '\n'; }
  }
  if (state_socket >= 0) close(state_socket);
  close(fd);
  if (sync_control_fd >= 0) close(sync_control_fd);
  std::cout << "{\"event\":\"closed\",\"force_n\":0.0}" << std::endl;
  return code;
}
