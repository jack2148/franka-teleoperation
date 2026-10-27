"""Test-only socketpair owner; no HDAPI imports, USB, state UDP or physical force."""
import argparse
import json
from pathlib import Path
import select
import signal
import socket
import struct
import time


def send_from_work(channel, token):
    """Spawn fixture imports no pytest and has no device access."""
    from geomagic.standalone.device import ForceSender
    sender = ForceSender(channel, token)
    sender.send((0, 0, 1))
    sender.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-fd', type=int)
    parser.add_argument('--state-port')
    parser.add_argument('--token')
    parser.add_argument('--parent-pid')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--sync-control-fd', type=int)
    args = parser.parse_args()
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    sock = socket.socket(fileno=args.force_fd)
    sock.setblocking(False)
    print(json.dumps({'event': 'device', 'model': 'FAKE', 'serial': 'NO-HARDWARE'}), flush=True)
    print(json.dumps({'event': 'ready'}), flush=True)
    def status():
        print(json.dumps({'event':'hdapi_status','servo_hz':1000,'state_hz':500,
                          'rendered_force_n':0.,'watchdog':'ZERO'}),flush=True)
    status()
    next_status=time.monotonic()+1
    packets = []
    def drain():
        while True:
            try: payload = sock.recv(256)
            except BlockingIOError: break
            packets.append(list(struct.unpack('<8sQQd3dII', payload)[1:]))
    while not stopped:
        select.select([sock], [], [], .01)
        drain()
        if time.monotonic() >= next_status:
            status()
            next_status=time.monotonic()+1
    drain()  # Explicit zero is sent before the parent requests termination.
    args.trace.write_text(json.dumps({'packets': packets,
        'events': ['zero_force', 'disable_force', 'close_device'], 'force_at_close': 0}))
    sock.close()

if __name__ == '__main__': main()
