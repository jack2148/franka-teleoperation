// Exercise the ACTUAL owner callback/lifecycle against a fake HD C API.
// Only libHDU math is linked. No libHD, USB, tty, scheduler or force hardware.
#define main teleop_geo_owner_main_unused
#include "../device/teleop_geo_hdapi.cpp"
#undef main
#include <cassert>
#include <vector>
#include <time.h>

namespace fake {
thread_local bool scheduler_thread = false;
int begins = 0, ends = 0, writes = 0;
bool in_frame = false, enabled = false;
HDerror read_error = HD_SUCCESS, end_error = HD_SUCCESS;
HDErrorInfo pending{};
Vec3 force{};
std::vector<std::string> events;
std::thread worker;
void tick() {
  scheduler_thread = true;
  begins = ends = writes = 0;
  in_frame = false;
  pending = {};
}
void reset() {
  tick();
  enabled = false;
  read_error = end_error = HD_SUCCESS;
  force = {};
  events.clear();
  stopping.store(false);
}
void check_one_frame() {
  assert(begins == 1 && ends == 1 && writes == 1 && !in_frame);
}
}

extern "C" {
HHD hdInitDevice(HDstring) { return 1; }
HHD hdGetCurrentDevice() { return 1; }
void hdBeginFrame(HHD) {
  assert(fake::scheduler_thread);  // No main-thread frames are allowed.
  if (fake::in_frame || ++fake::begins > 1) fake::pending.errorCode = HD_ILLEGAL_BEGIN;
  fake::in_frame = true;
}
void hdEndFrame(HHD) {
  assert(fake::in_frame);
  fake::in_frame = false;
  ++fake::ends;
  if (fake::end_error) fake::pending.errorCode = fake::end_error;
}
void hdGetDoublev(HDenum parameter, HDdouble* out) {
  assert(fake::in_frame);
  if (parameter == HD_CURRENT_POSITION) {
    out[0] = out[1] = out[2] = 0;
    if (fake::read_error) fake::pending.errorCode = fake::read_error;
  } else if (parameter == HD_CURRENT_TRANSFORM) {
    for (int i = 0; i < 16; ++i) out[i] = i % 5 == 0 ? 1 : 0;
  } else assert(false);
}
void hdGetIntegerv(HDenum parameter, HDint* out) {
  *out = parameter == HD_CALIBRATION_STYLE ? HD_CALIBRATION_INKWELL : HD_DEVICE_BUTTON_1;
}
void hdSetDoublev(HDenum parameter, const HDdouble* value) {
  assert(fake::scheduler_thread && fake::in_frame && parameter == HD_CURRENT_FORCE);
  ++fake::writes;
  fake::force = {value[0], value[1], value[2]};
}
HDErrorInfo hdGetError() { auto error = fake::pending; fake::pending = {}; return error; }
HDstring hdGetErrorString(HDerror) { return "fake HD error"; }
HDstring hdGetString(HDenum) { return "FAKE"; }
void hdEnable(HDenum parameter) {
  assert(fake::scheduler_thread && parameter == HD_FORCE_OUTPUT);
  fake::enabled = true;
  fake::events.push_back("enable");
}
void hdDisable(HDenum parameter) {
  assert(parameter == HD_FORCE_OUTPUT);
  fake::enabled = false;
  fake::events.push_back("disable");
}
HDboolean hdIsEnabled(HDenum) { return fake::enabled ? HD_TRUE : HD_FALSE; }
HDenum hdCheckCalibration() { return HD_CALIBRATION_OK; }
void hdUpdateCalibration(HDenum) {}
void hdStartScheduler() {}
void hdStopScheduler() { fake::events.push_back("stop"); }
HDSchedulerHandle hdScheduleAsynchronous(HDSchedulerCallback, void*, HDushort) { return 7; }
void hdScheduleSynchronous(HDSchedulerCallback, void*, HDushort) {
  assert(false && "A second zero-frame callback must never be scheduled");
}
void hdUnschedule(HDSchedulerHandle) {
  if (fake::worker.joinable()) fake::worker.join();
  fake::events.push_back("unschedule");
}
void hdDisableDevice(HHD) {
  assert(!fake::enabled);
  fake::events.push_back("close");
}
}

int main() {
  int tests = 0;
  {
    fake::reset();
    timespec clock_value{};
    clock_gettime(CLOCK_MONOTONIC, &clock_value);
    const auto linux_ns=static_cast<std::uint64_t>(clock_value.tv_sec)*1000000000+clock_value.tv_nsec;
    const auto steady_ns=geo_sync::now_ns();
    assert(steady_ns>=linux_ns && steady_ns-linux_ns<1000000);
    const std::string trace_dir="/tmp/native-sync-not-written-"+std::to_string(getpid());
    geo_sync::Trace trace(trace_dir, 1., 2);
    Shared state;
    state.sync=&trace;
    state.parent = getppid();
    state.publish({{1,0,0}, now_s(), 1, true},geo_sync::now_ns());
    assert(servo_callback(&state) == HD_CALLBACK_CONTINUE);
    fake::check_one_frame();
    assert(state.ready.load() && state.output_initialized && !state.fault.load());
    assert(norm(fake::force) == 0);  // Explicit first-frame zero even with a command.
    ++tests;
    fake::tick();
    state.previous_tick = now_s() - .001;
    state.publish({{1,0,0}, now_s(), 2, true},geo_sync::now_ns());
    assert(servo_callback(&state) == HD_CALLBACK_CONTINUE);
    fake::check_one_frame();
    assert(norm(fake::force) > 0 && norm(fake::force) <= kHardCapN);
    assert(trace.servo_count==2 && trace.servo_records[1].force_sequence==2);
    assert(trace.servo_records[1].received<=trace.servo_records[1].applied);
    assert(trace.servo_records[1].callback<=trace.servo_records[1].acquired);
    assert(trace.servo_records[1].acquired<=trace.servo_records[1].applied);
    assert(trace.servo_records[1].applied<=trace.servo_records[1].committed);
    assert(trace.servo_records[1].force==fake::force);
    assert(!std::filesystem::exists(trace_dir)); // No callback file I/O.
    ++tests;
    fake::tick();
    stopping.store(true);
    assert(servo_callback(&state) == HD_CALLBACK_DONE);
    fake::check_one_frame();
    assert(norm(fake::force) == 0 && state.stop_zero_written.load());
    assert(trace.servo_count==2 && trace.servo_limited);
    ++tests;
  }
  for (bool end_fault : {false, true}) {
    fake::reset();
    Shared state;
    state.parent = getppid();
    state.output_initialized = true;
    state.previous_tick = now_s() - .001;
    state.publish({{1,0,0}, now_s(), 1, true});
    fake::enabled = true;
    if (end_fault) fake::end_error = HD_COMM_ERROR;
    else fake::read_error = HD_COMM_ERROR;
    assert(servo_callback(&state) == HD_CALLBACK_DONE);
    fake::check_one_frame();
    assert(state.fault.load() && state.error_code.load() == HD_COMM_ERROR);
    assert(!fake::enabled && stopping.load());
    if (!end_fault) assert(norm(fake::force) == 0);
    ++tests;
  }
  {
    fake::reset();
    Shared state;
    state.parent = getppid();
    state.output_initialized = true;
    fake::enabled = true;
    Device device;
    device.handle = 1; device.callback = 7; device.scheduler = true; device.shared = &state;
    fake::scheduler_thread = false;
    // Main requests stop; the actual sole callback acknowledges zero next tick.
    fake::worker = std::thread([&] {
      while (!stopping.load()) std::this_thread::yield();
      fake::tick();
      assert(servo_callback(&state) == HD_CALLBACK_DONE);
      fake::check_one_frame();
    });
    zero_then_close(device);
    assert(state.stop_zero_written.load() && device.closed && norm(fake::force) == 0);
    assert((fake::events == std::vector<std::string>{"disable","unschedule","stop","disable","close"}));
    ++tests;
  }
  {
    fake::reset();
    Shared state;
    state.fault.store(true); // A failed scheduler must not block a shutdown frame.
    Device device;
    device.handle = 1; device.callback = 7; device.scheduler = true; device.shared = &state;
    fake::scheduler_thread = false;
    zero_then_close(device);
    assert(device.closed && fake::begins == 0);
    assert((fake::events == std::vector<std::string>{"disable","unschedule","stop","disable","close"}));
    ++tests;
  }
  {
    fake::reset();
    Shared state;
    state.parent = getppid() + 100; // Parent loss requests zero in the same frame.
    assert(servo_callback(&state) == HD_CALLBACK_DONE);
    fake::check_one_frame();
    assert(norm(fake::force) == 0 && state.stop_zero_written.load());
    assert(!fake::enabled && stopping.load());
    ++tests;
  }
  std::cout << "teleop_geo native frame tests: " << tests << " passed, 0 failed, 0 skipped\n";
}
