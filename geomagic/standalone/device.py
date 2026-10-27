"""Preflight and lifecycle for one dedicated state+force HDAPI child."""
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import threading
import time

from .haptics import force_packet
from geomagic.teleop.sync_capture import get_sync_capture

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / 'geomagic/device/bin/teleop_geo_hdapi'
OWNER_NAMES = {'hdapi_state_stream', 'hdapi_haptic_servo', 'hdapi_force_diagnostic',
               'teleop_geo_hdapi', 'Touch_Setup', 'Touch_Diagnostic', 'omni_state',
               'inspect_openhaptics'}


def runtime_environment(runtime):
    env = os.environ.copy()
    env['GEOMAGIC_LOCAL'] = str(runtime)
    env['GTDD_HOME'] = str(runtime / '3DSystems')
    libraries = [runtime / 'lib', runtime / 'compat/lib/x86_64-linux-gnu',
                 runtime / 'compat/usr/lib/x86_64-linux-gnu']
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, libraries)) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def other_owners(proc_root=Path('/proc'), exclude=None):
    """Conservative, read-only scan including generic libHD users and tty owners."""
    exclude = set(exclude or ()) | {os.getpid()}
    owners = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) in exclude:
            continue
        try:
            argv = (process / 'cmdline').read_bytes().split(b'\0')
            name = Path(os.fsdecode(argv[0])).name if argv and argv[0] else ''
        except OSError:
            continue
        try:
            maps = (process / 'maps').read_text(errors='replace')
        except OSError:
            maps = ''  # A named owner remains a conflict even if maps is private.
        reasons = []
        if name in OWNER_NAMES: reasons.append(name)
        if 'libHD.so' in maps or 'libPhantomIOLib' in maps: reasons.append('OpenHaptics library loaded')
        try:
            if any(os.readlink(fd) == '/dev/ttyACM0' for fd in (process / 'fd').iterdir()):
                reasons.append('/dev/ttyACM0 open')
        except OSError:
            pass
        if reasons:
            owners.append((int(process.name), ', '.join(reasons)))
    return owners


def preflight(*, runtime=None, state_port=5568, sys_root=Path('/sys/bus/usb/devices'),
              tty=Path('/dev/ttyACM0'), proc_root=Path('/proc'), binary=BINARY):
    runtime = Path(runtime or os.environ.get('GEOMAGIC_LOCAL', str(Path.home()/'.local/geomagic')))
    owners = other_owners(proc_root)
    if owners:
        raise RuntimeError('Touch already has a possible owner: ' + '; '.join(f'{p}: {n}' for p, n in owners) + '. Close that application manually.')
    present = False
    for device in sys_root.iterdir():
        try:
            present |= ((device / 'idVendor').read_text().strip(),
                        (device / 'idProduct').read_text().strip()) == ('2988', '0302')
        except OSError:
            pass
    if not present: raise RuntimeError('Geomagic Touch USB 2988:0302 not present')
    if not tty.exists() or not stat.S_ISCHR(tty.stat().st_mode) or not os.access(tty, os.R_OK | os.W_OK):
        raise RuntimeError(f'{tty} must be an accessible read/write character device')
    for relative in ('include/HD/hd.h', 'lib/libHD.so', 'lib/libHDU.a', '3DSystems'):
        if not (runtime / relative).exists(): raise RuntimeError(f'OpenHaptics runtime missing: {runtime / relative}')
    if not os.access(binary, os.X_OK):
        raise RuntimeError(f'Build the dedicated binary first: bash {ROOT}/geomagic/device/build_teleop_geo_hdapi.sh')
    libraries = subprocess.run(['ldd', str(binary)], env=runtime_environment(runtime),
                               capture_output=True, text=True, timeout=5)
    if libraries.returncode or 'not found' in libraries.stdout + libraries.stderr:
        raise RuntimeError('Dedicated HDAPI binary libraries do not resolve: ' + libraries.stdout + libraries.stderr)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try: probe.bind(('127.0.0.1', state_port))
        except OSError as error: raise RuntimeError(f'UDP state port {state_port} is occupied') from error
    return runtime


class OwnerLock:
    """Prevent two standalone launchers racing; existing owners are scanned too."""
    def __init__(self, path=None):
        self.path = Path(path or f'/tmp/teleop_geo-owner-{os.getuid()}.lock')
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError('Another teleop_geo launcher holds the owner lock')
        return self

    def __exit__(self, *args):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class ForceSender:
    """Shared nonblocking 64-byte wire sender; UI workers inherit only this endpoint.

    UI sends use host monotonic nanoseconds for strictly increasing sequences
    across successive spawned WORK workers and the parent's shutdown zero.
    teleop_geo retains its original incrementing sequence.
    """
    def __init__(self, sender, token, *, sequence_clock=True):
        self.sender, self.token = sender, token
        self.sender.setblocking(False)
        self.sequence = 0
        self.sequence_clock = sequence_clock
        self.sync = get_sync_capture()
        self.transport_error = None
        self.closed = False

    def send(self, force, enabled=True, *, diagnostics=None, sim_time=None):
        self.sequence = time.monotonic_ns() if self.sequence_clock else self.sequence + 1
        data = force_packet(self.token, self.sequence, time.monotonic(), force, enabled=enabled)
        trace = self.sync
        sent_ns = time.monotonic_ns() if trace else None
        try:
            sent = self.sender.send(data) == len(data)
            if trace:
                trace.emit('force', sim_time, H2_ns=sent_ns, force_session=str(self.token),
                           force_sequence=self.sequence, sent=sent, enabled=enabled,
                           **(diagnostics or {'device_force': list(force)}))
            return sent
        except OSError as error:
            self.transport_error = error
            return False  # Let the session preserve the HDAPI failure diagnosis.

    def close(self):
        if self.closed:
            return
        try:
            self.send((0, 0, 0), enabled=False)
        finally:
            self.sender.close()
            self.closed = True


class HapticOwner(ForceSender):
    """Private datagram channel: no force UDP port and no competing streamer."""
    def __init__(self, runtime, state_port=5568, *, command=None, sequence_clock=False, capture_control=False):
        self.runtime = Path(runtime)
        self.state_port = state_port
        self.command = list(command) if command is not None else [str(BINARY)]
        self.sync = get_sync_capture()
        self.sender, self.receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sender.setblocking(False)
        self.receiver.setblocking(False)
        self.token = int.from_bytes(os.urandom(8), 'little') or 1
        self.sequence = 0
        self.sequence_clock = sequence_clock
        self.capture_channel = self.capture_receiver = None
        if capture_control:
            self.capture_channel, self.capture_receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.capture_channel.settimeout(30.)
        self.process = None
        self.thread = None
        self.ready = threading.Event()
        self.info = {}
        self.status = {}
        self.last_error = None
        self.transport_error = None
        self.closed = False
        self.sent_zero_before_close = False

    def start(self):
        if self.process is not None: raise RuntimeError('one owner may only start once')
        args = self.command + ['--force-fd', str(self.receiver.fileno()),
            '--state-port', str(self.state_port), '--token', str(self.token), '--parent-pid', str(os.getpid())]
        fds = (self.receiver.fileno(),)
        if self.capture_receiver is not None:
            args.extend(['--sync-control-fd', str(self.capture_receiver.fileno())])
            fds += (self.capture_receiver.fileno(),)
        self.process = subprocess.Popen(args, pass_fds=fds,
            env=runtime_environment(self.runtime), stdout=subprocess.PIPE, text=True, bufsize=1)
        self.receiver.close()
        if self.capture_receiver is not None:
            self.capture_receiver.close()
        self.thread = threading.Thread(target=self._read_status, name='teleop_geo_hdapi_status', daemon=True)
        self.thread.start()

    def _read_status(self):
        for line in self.process.stdout:
            try: message = json.loads(line)
            except (ValueError, TypeError): continue
            if message.get('event') == 'device':
                self.info = message
                print(f"Touch: {message.get('model')} / serial {message.get('serial')}", flush=True)
            elif message.get('event') == 'ready': self.ready.set()
            elif message.get('event') == 'hdapi_status':
                self.status = {**message, 'received_at': time.monotonic()}
            elif message.get('event') == 'calibration': print(message.get('instruction', 'Calibration'), flush=True)
            elif message.get('event') == 'error':
                self.last_error = message
                print(f"HDAPI error: {message.get('description')} / code {message.get('code')} / internal {message.get('internal_code')}", flush=True)

    def exit_description(self):
        # Drain the short final diagnostic after process exit, not in the servo.
        if self.thread is not None: self.thread.join(timeout=.1)
        code = self.process.poll()
        detail = f'code {code}' if code is not None else 'exit pending'
        if self.last_error:
            detail += f"; {self.last_error.get('description')}, HD {self.last_error.get('code')}, internal {self.last_error.get('internal_code')}"
        return detail

    def send_failure_description(self):
        # Disconnection can precede waitpid observing exit. Drain the child's
        # final diagnostic briefly, only on failure, then stop the session.
        if self.thread is not None: self.thread.join(timeout=.1)
        if self.last_error or (self.process is not None and self.process.poll() is not None):
            return 'HDAPI owner stopped (' + self.exit_description() + ')'
        return 'force command channel failed: ' + str(self.transport_error or 'short send')

    def wait_ready(self, timeout=40, *, stop_requested=lambda: False):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stop_requested(): raise KeyboardInterrupt
            if self.process.poll() is not None: raise RuntimeError(f'HDAPI owner exited ({self.exit_description()})')
            if self.ready.wait(.05): return
        raise RuntimeError('HDAPI startup/calibration timed out')

    def close(self):
        if self.closed: return
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.sent_zero_before_close = self.send((0, 0, 0), enabled=False)
                except OSError:
                    pass  # Independent servo watchdog and SIGTERM safety gate.
                self.process.terminate()  # Our single owned child; C++ zeros before closing.
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()  # Only this owned child; output also has the parent watchdog.
                    self.process.wait(timeout=2)
        finally:
            self.sender.close()
            self.receiver.close()
            for channel in (self.capture_channel, self.capture_receiver):
                if channel is not None:
                    channel.close()
            if self.thread is not None: self.thread.join(timeout=1)
            if (self.process is not None and self.process.stdout is not None and
                    (self.thread is None or not self.thread.is_alive())):
                self.process.stdout.close()
            self.closed = True

    def __enter__(self):
        try:
            self.start()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *args):
        self.close()
