// Force-disabled Geomagic Touch state streamer: HDAPI -> Phase-1 UDP (localhost).
//
// Device acquisition (init, scheduler, inkwell calibration gate, position,
// transform/quaternion and button reads, error checks) is reused from the
// validated reference:
//   geomagic_franka_teleop_import_v2/src/haptics/hdapi_haptic_servo.cpp
// Nothing force related is reused. This program never enables HD_FORCE_OUTPUT,
// never writes HD_CURRENT_FORCE. The optional inherited diagnostic control
// socket receives only START/SAVE requests from the UI supervisor.
//
// Units: HD_CURRENT_POSITION is native millimetres (validated holder pose
// [0, -65.51, -88.11] mm); it is converted exactly once (x 0.001) to metres.
// Orientation is the raw device-frame quaternion from HD_CURRENT_TRANSFORM,
// emitted as [qx, qy, qz, qw]. No FR3/application mapping is applied here.
#include <HD/hd.h>
#include "sync_trace.h"
#include <HDU/hduError.h>
#include <HDU/hduMatrix.h>
#include <HDU/hduQuaternion.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using Clock = std::chrono::steady_clock;  // CLOCK_MONOTONIC on Linux
static_assert(std::atomic<double>::is_always_lock_free, "streamer requires lock-free atomics");
static_assert(std::atomic<unsigned long long>::is_always_lock_free, "streamer requires lock-free atomics");

namespace {

constexpr double kMillimetresToMetres = 0.001;
constexpr const char* kClockDomain = "server.monotonic.hdapi";
constexpr unsigned long long kMaxConsecutiveDeviceErrors = 50;

std::atomic<bool> stop_requested{false};

void signal_handler(int) { stop_requested.store(true, std::memory_order_relaxed); }

double monotonic_s() {
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

const char* safe_string(const char* value) { return value ? value : "UNAVAILABLE"; }

const char* calibration_name(HDenum state) {
  switch (state) {
    case HD_CALIBRATION_OK: return "OK";
    case HD_CALIBRATION_NEEDS_UPDATE: return "NEEDS_UPDATE";
    case HD_CALIBRATION_NEEDS_MANUAL_INPUT: return "NEEDS_MANUAL_INPUT";
    default: return "UNKNOWN";
  }
}

struct Options {
  int port{5558};
  double rate_hz{500.0};
  double duration_s{0.0};  // 0 = until SIGINT/SIGTERM
  double calibration_timeout_s{30.0};
  double status_interval_s{1.0};  // 0 = no periodic status lines
  std::string session_id;
  int sync_control_fd{-1};  // private socket inherited from the UI supervisor
};

bool valid_identifier(const std::string& value) {
  if (value.empty() || value.size() > 128 || !std::isalnum(static_cast<unsigned char>(value[0])))
    return false;
  return std::all_of(value.begin(), value.end(), [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == ':' || c == '-';
  });
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    auto value = [&]() -> std::string {
      if (index + 1 >= argc) throw std::runtime_error("missing value for " + key);
      return argv[++index];
    };
    if (key == "--port") options.port = std::stoi(value());
    else if (key == "--rate-hz") options.rate_hz = std::stod(value());
    else if (key == "--duration") options.duration_s = std::stod(value());
    else if (key == "--calibration-timeout") options.calibration_timeout_s = std::stod(value());
    else if (key == "--status-interval") options.status_interval_s = std::stod(value());
    else if (key == "--session-id") options.session_id = value();
    else if (key == "--sync-control-fd") {
      options.sync_control_fd = std::stoi(value());
      if (options.sync_control_fd < 0) throw std::runtime_error("sync-control-fd must be >= 0");
    }
    else throw std::runtime_error("unknown option: " + key);
  }
  if (options.port <= 0 || options.port > 65535) throw std::runtime_error("port must be 1..65535");
  if (!std::isfinite(options.rate_hz) || options.rate_hz <= 0.0 || options.rate_hz > 1000.0)
    throw std::runtime_error("rate-hz must be in (0, 1000]");
  if (!std::isfinite(options.duration_s) || options.duration_s < 0.0)
    throw std::runtime_error("duration must be >= 0");
  if (!std::isfinite(options.calibration_timeout_s) || options.calibration_timeout_s <= 0.0)
    throw std::runtime_error("calibration-timeout must be > 0");
  if (!std::isfinite(options.status_interval_s) || options.status_interval_s < 0.0)
    throw std::runtime_error("status-interval must be >= 0");
  if (options.session_id.empty()) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
    options.session_id = "hdapi-" + std::to_string(getpid()) + "-" + std::to_string(ns);
  }
  if (!valid_identifier(options.session_id))
    throw std::runtime_error("session-id must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,127}");
  return options;
}

// Same gate as the validated servo: never bypass inkwell calibration.
bool calibrate_device(HDenum style, double timeout_s) {
  const auto deadline = Clock::now() + std::chrono::duration<double>(timeout_s);
  bool prompted = false;
  while (Clock::now() < deadline && !stop_requested.load()) {
    const HDenum state = hdCheckCalibration();
    if (state == HD_CALIBRATION_OK) return true;
    if (state == HD_CALIBRATION_NEEDS_UPDATE) {
      hdUpdateCalibration(style);
      const HDErrorInfo error = hdGetError();
      if (HD_DEVICE_ERROR(error)) {
        hduPrintError(stderr, &error, "hdUpdateCalibration");
        return false;
      }
    } else if (state == HD_CALIBRATION_NEEDS_MANUAL_INPUT) {
      if (!prompted) {
        std::cout << "calibration_action=place stylus fully in the inkwell/holder" << std::endl;
        prompted = true;
      }
    } else {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return false;
}

// Latest device sample, written only by the HDAPI scheduler thread and read by
// the publisher through a seqlock (odd version = write in progress). Every
// field is a lock-free atomic, so a torn read is detected, never undefined.
struct SharedState {
  geo_sync::Trace* sync{};
  std::atomic<unsigned long long> version{0};
  std::atomic<unsigned long long> sample_index{0};  // valid samples observed
  std::atomic<double> sample_time_s{0.0};
  std::array<std::atomic<double>, 3> position_m{};
  std::array<std::atomic<double>, 4> orientation_xyzw{};
  std::atomic<int> buttons{0};

  std::atomic<unsigned long long> callbacks{0};
  std::atomic<unsigned long long> valid_callbacks{0};
  std::atomic<unsigned long long> device_errors{0};
  std::atomic<unsigned long long> calibration_invalid{0};
  std::atomic<unsigned long long> nonfinite_samples{0};
  std::atomic<unsigned long long> force_output_observed_enabled{0};
  std::atomic<int> last_error_code{0};
  std::atomic<double> max_callback_gap_s{0.0};
  std::atomic<bool> scheduler_fault{false};

  // Scheduler-thread-only bookkeeping.
  double previous_callback_s{0.0};
  unsigned long long consecutive_errors{0};
};

struct Snapshot {
  unsigned long long sample_index{0};
  double sample_time_s{0.0};
  std::array<double, 3> position_m{};
  std::array<double, 4> orientation_xyzw{};
  int buttons{0};
};

// Realtime callback: read state, check errors, store the latest sample. No
// allocation, I/O, serialization, socket use or force write happens here.
HDCallbackCode HDCALLBACK state_callback(void* user_data) {
  auto* state = static_cast<SharedState*>(user_data);
  const double now_s = monotonic_s();
  if (state->previous_callback_s > 0.0) {
    const double gap = now_s - state->previous_callback_s;
    if (gap > state->max_callback_gap_s.load(std::memory_order_relaxed))
      state->max_callback_gap_s.store(gap, std::memory_order_relaxed);
  }
  state->previous_callback_s = now_s;
  state->callbacks.fetch_add(1, std::memory_order_relaxed);

  const bool calibration_valid = hdCheckCalibration() == HD_CALIBRATION_OK;
  HDdouble position_mm[3]{};
  hduMatrix transform;
  HDint buttons = 0;
  hdBeginFrame(hdGetCurrentDevice());
  hdGetDoublev(HD_CURRENT_POSITION, position_mm);
  hdGetDoublev(HD_CURRENT_TRANSFORM, transform);
  hdGetIntegerv(HD_CURRENT_BUTTONS, &buttons);
  const auto acquired_ns = state->sync ? geo_sync::now_ns() : 0;
  const bool force_enabled = hdIsEnabled(HD_FORCE_OUTPUT) == HD_TRUE;
  hdEndFrame(hdGetCurrentDevice());
  const HDErrorInfo error = hdGetError();

  if (force_enabled) {
    // Must never happen; disable defensively and stop the stream.
    state->force_output_observed_enabled.fetch_add(1, std::memory_order_relaxed);
    hdDisable(HD_FORCE_OUTPUT);
    state->scheduler_fault.store(true);
    return HD_CALLBACK_DONE;
  }
  if (HD_DEVICE_ERROR(error)) {
    state->device_errors.fetch_add(1, std::memory_order_relaxed);
    state->last_error_code.store(static_cast<int>(error.errorCode), std::memory_order_relaxed);
    if (++state->consecutive_errors >= kMaxConsecutiveDeviceErrors) {
      state->scheduler_fault.store(true);
      return HD_CALLBACK_DONE;
    }
    return HD_CALLBACK_CONTINUE;
  }
  state->consecutive_errors = 0;
  if (!calibration_valid) {
    state->calibration_invalid.fetch_add(1, std::memory_order_relaxed);
    return HD_CALLBACK_CONTINUE;
  }

  // Validated orientation extraction (identical to the reference servo).
  hduMatrix rotation(transform);
  rotation.getRotationMatrix(rotation);
  const hduQuaternion quaternion(rotation);
  const double q[4]{quaternion.v()[0], quaternion.v()[1], quaternion.v()[2], quaternion.s()};
  const double p[3]{position_mm[0] * kMillimetresToMetres, position_mm[1] * kMillimetresToMetres,
                    position_mm[2] * kMillimetresToMetres};
  bool finite = true;
  for (double value : p) finite = finite && std::isfinite(value);
  for (double value : q) finite = finite && std::isfinite(value);
  if (!finite) {
    state->nonfinite_samples.fetch_add(1, std::memory_order_relaxed);
    return HD_CALLBACK_CONTINUE;
  }

  state->version.fetch_add(1, std::memory_order_acq_rel);  // odd: writing
  for (int axis = 0; axis < 3; ++axis) state->position_m[axis].store(p[axis], std::memory_order_relaxed);
  for (int i = 0; i < 4; ++i) state->orientation_xyzw[i].store(q[i], std::memory_order_relaxed);
  state->buttons.store(buttons, std::memory_order_relaxed);
  state->sample_time_s.store(now_s, std::memory_order_relaxed);
  state->sample_index.fetch_add(1, std::memory_order_relaxed);
  state->version.fetch_add(1, std::memory_order_release);  // even: complete
  state->valid_callbacks.fetch_add(1, std::memory_order_relaxed);
  if (state->sync) state->sync->servo({geo_sync::ns(now_s), acquired_ns, 0, 0, 0, 0,
      state->sample_index.load(), 0, {p[0],p[1],p[2]}, {}, {}, {q[0],q[1],q[2],q[3]}, buttons, false, true});
  return HD_CALLBACK_CONTINUE;
}

bool read_snapshot(const SharedState& state, Snapshot& out) {
  for (int attempt = 0; attempt < 8; ++attempt) {
    const auto before = state.version.load(std::memory_order_acquire);
    if (before & 1ULL) continue;
    out.sample_index = state.sample_index.load(std::memory_order_relaxed);
    out.sample_time_s = state.sample_time_s.load(std::memory_order_relaxed);
    for (int axis = 0; axis < 3; ++axis) out.position_m[axis] = state.position_m[axis].load(std::memory_order_relaxed);
    for (int i = 0; i < 4; ++i) out.orientation_xyzw[i] = state.orientation_xyzw[i].load(std::memory_order_relaxed);
    out.buttons = state.buttons.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (state.version.load(std::memory_order_relaxed) == before) return true;
  }
  return false;
}

struct PublishMetrics {
  unsigned long long published{0};
  unsigned long long send_errors{0};
  unsigned long long skipped_duplicate{0};
  unsigned long long invalid_pose{0};
  unsigned long long torn_reads{0};
  unsigned long long ticks{0};
};

// Phase-1 protocol v1 packet. The quaternion is normalized and given w >= 0
// (same canonicalization as the validated run_haptic_mujoco parser); this
// does not change the frame. Returns false for an unusable quaternion.
int format_packet(char* buffer, std::size_t size, const Options& options,
                  unsigned long long sequence, const Snapshot& sample) {
  std::array<double, 4> q = sample.orientation_xyzw;
  const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!std::isfinite(norm) || std::fabs(norm - 1.0) > 0.05) return -1;
  for (double& value : q) value /= norm;
  if (q[3] < 0.0)
    for (double& value : q) value = -value;
  const bool grey = (sample.buttons & HD_DEVICE_BUTTON_1) != 0;
  const bool white = (sample.buttons & HD_DEVICE_BUTTON_2) != 0;
  return std::snprintf(
      buffer, size,
      "{\"grey_button\":%s,\"orientation_xyzw\":[%.12f,%.12f,%.12f,%.12f],"
      "\"position_m\":[%.9f,%.9f,%.9f],\"protocol_version\":1,\"sequence\":%llu,"
      "\"session_id\":\"%s\",\"source_clock_domain\":\"%s\",\"source_timestamp\":%.9f,"
      "\"valid\":true,\"white_button\":%s}",
      grey ? "true" : "false", q[0], q[1], q[2], q[3],
      sample.position_m[0], sample.position_m[1], sample.position_m[2], sequence,
      options.session_id.c_str(), kClockDomain, sample.sample_time_s, white ? "true" : "false");
}

void print_status(const SharedState& state, const PublishMetrics& metrics, double elapsed,
                  const Snapshot& last) {
  std::printf(
      "status t=%.1f callbacks=%llu valid=%llu errors=%llu published=%llu dup_skipped=%llu "
      "invalid_pose=%llu grey=%d white=%d pos_mm=[%.1f,%.1f,%.1f]\n",
      elapsed, state.callbacks.load(), state.valid_callbacks.load(), state.device_errors.load(),
      metrics.published, metrics.skipped_duplicate, metrics.invalid_pose,
      (last.buttons & HD_DEVICE_BUTTON_1) ? 1 : 0, (last.buttons & HD_DEVICE_BUTTON_2) ? 1 : 0,
      last.position_m[0] * 1000.0, last.position_m[1] * 1000.0, last.position_m[2] * 1000.0);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::unique_ptr<geo_sync::Trace> sync;
  try {
    options = parse_options(argc, argv);
    sync = geo_sync::Trace::from_environment();
    if (sync) {
      sync->session = options.session_id;
      if (options.sync_control_fd >= 0) sync->wait_for_begin();
    } else if (options.sync_control_fd >= 0) {
      throw std::runtime_error("sync-control-fd requires GEOMAGIC_SYNC_CAPTURE");
    }
  } catch (const std::exception& error) {
    std::cerr << "configuration_error=" << error.what() << '\n';
    return 64;
  }
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  HHD device = hdInitDevice(HD_DEFAULT_DEVICE);
  HDErrorInfo error = hdGetError();
  if (HD_DEVICE_ERROR(error)) {
    hduPrintError(stderr, &error, "hdInitDevice");
    return 2;
  }
  hdDisable(HD_FORCE_OUTPUT);  // invariant: force output is never enabled
  std::cout << "hdInitDevice=OK\n"
            << "model=" << safe_string(hdGetString(HD_DEVICE_MODEL_TYPE)) << '\n'
            << "vendor=" << safe_string(hdGetString(HD_DEVICE_VENDOR)) << '\n'
            << "serial=" << safe_string(hdGetString(HD_DEVICE_SERIAL_NUMBER)) << '\n'
            << "FORCE_OUTPUT_ENABLED=" << (hdIsEnabled(HD_FORCE_OUTPUT) ? "YES" : "NO") << std::endl;
  if (hdIsEnabled(HD_FORCE_OUTPUT)) {
    std::cerr << "fatal=force output enabled after hdDisable\n";
    hdDisableDevice(device);
    return 10;
  }

  HDint calibration_styles = 0;
  hdGetIntegerv(HD_CALIBRATION_STYLE, &calibration_styles);
  HDenum calibration_style = HD_CALIBRATION_AUTO;
  if (calibration_styles & HD_CALIBRATION_INKWELL) calibration_style = HD_CALIBRATION_INKWELL;
  else if (calibration_styles & HD_CALIBRATION_ENCODER_RESET)
    calibration_style = HD_CALIBRATION_ENCODER_RESET;

  hdStartScheduler();
  error = hdGetError();
  if (HD_DEVICE_ERROR(error)) {
    hduPrintError(stderr, &error, "hdStartScheduler");
    hdDisableDevice(device);
    return 3;
  }
  std::cout << "scheduler=STARTED\n"
            << "calibration_styles_mask=" << calibration_styles << '\n'
            << "calibration_before=" << calibration_name(hdCheckCalibration()) << std::endl;
  if (!calibrate_device(calibration_style, options.calibration_timeout_s)) {
    std::cerr << "calibration_after=" << calibration_name(hdCheckCalibration()) << '\n';
    hdStopScheduler();
    hdDisableDevice(device);
    return 4;
  }
  std::cout << "calibration_after=OK" << std::endl;

  SharedState state;
  state.sync = sync.get();
  HDSchedulerHandle callback =
      hdScheduleAsynchronous(state_callback, &state, HD_MAX_SCHEDULER_PRIORITY);
  error = hdGetError();
  if (HD_DEVICE_ERROR(error)) {
    hduPrintError(stderr, &error, "hdScheduleAsynchronous");
    hdStopScheduler();
    hdDisableDevice(device);
    return 5;
  }

  const int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) {
    std::cerr << "socket_error=" << std::strerror(errno) << '\n';
    hdStopScheduler();
    hdUnschedule(callback);
    hdDisableDevice(device);
    return 6;
  }
  sockaddr_in destination{};
  destination.sin_family = AF_INET;
  destination.sin_port = htons(static_cast<uint16_t>(options.port));
  destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // localhost only

  std::cout << "destination=127.0.0.1:" << options.port << '\n'
            << "protocol_version=1\n"
            << "session_id=" << options.session_id << '\n'
            << "source_clock_domain=" << kClockDomain << '\n'
            << "publish_rate_hz=" << options.rate_hz << '\n'
            << "position_units=native_mm*0.001->m\n"
            << "FORCE_OUTPUT_ENABLED=" << (hdIsEnabled(HD_FORCE_OUTPUT) ? "YES" : "NO") << '\n'
            << "streaming=STARTED" << std::endl;

  PublishMetrics metrics;
  Snapshot sample;
  Snapshot last_sent;
  unsigned long long last_sample_index = 0;
  unsigned long long sequence = 0;
  char packet[512];
  const auto period = std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(1.0 / options.rate_hz));
  const auto start = Clock::now();
  const double start_s = monotonic_s();
  auto next_tick = start;
  auto next_status = start + std::chrono::duration_cast<Clock::duration>(
                                 std::chrono::duration<double>(options.status_interval_s));
  const bool bounded = options.duration_s > 0.0;
  const auto deadline = start + std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(options.duration_s));

  while (!stop_requested.load() && !state.scheduler_fault.load() &&
         (!bounded || Clock::now() < deadline)) {
    // Control commands run in the publisher thread, never in the servo.
    // SAVE is requested only after the WORK control loop has stopped.
    if (options.sync_control_fd >= 0) {
      char request[4096];
      const auto length = recv(options.sync_control_fd, request, sizeof(request), MSG_DONTWAIT);
      if (length > 0) {
        std::string reply;
        try {
          const std::string command(request, static_cast<std::size_t>(length));
          if (command.rfind("BEGIN\t", 0) == 0) sync->begin(command.substr(6));
          else if (command == "SAVE") sync->checkpoint();
          else throw std::runtime_error("unknown capture command");
          reply = "OK\t" + sync->directory;
        } catch (const std::exception& e) { reply = "ERROR\t" + std::string(e.what()); }
        send(options.sync_control_fd, reply.data(), reply.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
      }
    }
    next_tick += period;
    std::this_thread::sleep_until(next_tick);
    const auto now = Clock::now();
    if (now - next_tick > 5 * period) next_tick = now;  // resync after a long stall
    ++metrics.ticks;
    if (!read_snapshot(state, sample)) {
      ++metrics.torn_reads;
      continue;
    }
    if (sample.sample_index == 0) continue;  // no valid device sample yet
    if (sample.sample_index == last_sample_index) {
      ++metrics.skipped_duplicate;  // never republish an old device sample as new
      continue;
    }
    last_sample_index = sample.sample_index;
    const int length = format_packet(packet, sizeof(packet), options, sequence, sample);
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(packet)) {
      ++metrics.invalid_pose;
      continue;
    }
    if (sendto(sock, packet, static_cast<std::size_t>(length), MSG_DONTWAIT,
               reinterpret_cast<sockaddr*>(&destination), sizeof(destination)) == length) {
      ++metrics.published;
      if (sync) sync->packet({geo_sync::now_ns(), sequence, sample.sample_index, geo_sync::ns(sample.sample_time_s)});
      ++sequence;  // strictly increasing over published packets
      last_sent = sample;
    } else {
      ++metrics.send_errors;
    }
    if (options.status_interval_s > 0.0 && now >= next_status) {
      print_status(state, metrics, std::chrono::duration<double>(now - start).count(), last_sent);
      next_status += std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(options.status_interval_s));
    }
  }

  const double end_s = monotonic_s();
  const bool fault = state.scheduler_fault.load();
  // Flush before any SDK teardown: an SDK shutdown failure cannot discard T0.
  if (sync) {
    try { sync->checkpoint(); }
    catch (const std::exception& e) { std::cerr << "synchronization report: " << e.what() << '\n'; }
  }
  hdStopScheduler();
  hdUnschedule(callback);
  hdDisable(HD_FORCE_OUTPUT);
  close(sock);
  hdDisableDevice(device);
  const HDErrorInfo shutdown_error = hdGetError();
  if (options.sync_control_fd >= 0) close(options.sync_control_fd);

  const double elapsed = std::max(1e-9, end_s - start_s);
  std::printf(
      "summary_json={\"session_id\":\"%s\",\"start_monotonic_s\":%.6f,\"end_monotonic_s\":%.6f,"
      "\"elapsed_s\":%.6f,\"hdapi_callbacks\":%llu,\"valid_callbacks\":%llu,"
      "\"device_errors\":%llu,\"last_error_code\":%d,\"calibration_invalid_callbacks\":%llu,"
      "\"nonfinite_samples\":%llu,\"max_callback_gap_ms\":%.3f,\"udp_published\":%llu,"
      "\"udp_send_errors\":%llu,\"skipped_duplicate_samples\":%llu,\"invalid_pose\":%llu,"
      "\"torn_reads\":%llu,\"publisher_ticks\":%llu,\"hdapi_callback_hz\":%.3f,"
      "\"valid_sample_hz\":%.3f,\"udp_publication_hz\":%.3f,"
      "\"force_output_observed_enabled\":%llu,\"force_output_enabled\":false,"
      "\"scheduler_fault\":%s,\"shutdown_error\":%s}\n",
      options.session_id.c_str(), start_s, end_s, elapsed, state.callbacks.load(),
      state.valid_callbacks.load(), state.device_errors.load(), state.last_error_code.load(),
      state.calibration_invalid.load(), state.nonfinite_samples.load(),
      state.max_callback_gap_s.load() * 1000.0, metrics.published, metrics.send_errors,
      metrics.skipped_duplicate, metrics.invalid_pose, metrics.torn_reads, metrics.ticks,
      state.callbacks.load() / elapsed, state.valid_callbacks.load() / elapsed,
      metrics.published / elapsed, state.force_output_observed_enabled.load(),
      fault ? "true" : "false", HD_DEVICE_ERROR(shutdown_error) ? "true" : "false");
  std::printf("FORCE_OUTPUT_ENABLED=NO\n");
  std::fflush(stdout);
  return fault ? 9 : 0;
}
