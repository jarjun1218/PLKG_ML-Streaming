import socket
import struct
import threading
import time


SYNC_REQ = b"T"
SYNC_RESP = b"R"
SYNC_PACKET = struct.Struct("!cId")
SYNC_RESPONSE_PACKET = struct.Struct("!cIddd")


class ClockSyncServer:
    def __init__(self, bind_ip="0.0.0.0", port=5006):
        self.bind_ip = bind_ip
        self.port = port
        self._thread = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._bind_error = None

    def start(self, wait_timeout=1.0):
        if self._thread is not None and self._thread.is_alive():
            self._ready.wait(timeout=wait_timeout)
            return
        self._stop.clear()
        self._ready.clear()
        self._bind_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=wait_timeout)
        if self._bind_error is not None:
            raise OSError(
                f"Clock sync server failed to bind {self.bind_ip}:{self.port}"
            ) from self._bind_error

    def stop(self):
        self._stop.set()

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.bind_ip, self.port))
            self._ready.set()
            sock.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(1024)
                except socket.timeout:
                    continue
                except OSError:
                    break

                if len(data) != SYNC_PACKET.size:
                    continue

                try:
                    msg_type, seq, t1 = SYNC_PACKET.unpack(data)
                except struct.error:
                    continue

                if msg_type != SYNC_REQ:
                    continue

                t2 = time.time()
                t3 = time.time()
                payload = SYNC_RESPONSE_PACKET.pack(SYNC_RESP, seq, t1, t2, t3)
                try:
                    sock.sendto(payload, addr)
                except OSError:
                    continue
        except OSError as exc:
            self._bind_error = exc
            self._ready.set()
        finally:
            sock.close()


def estimate_clock_offset(server_ip, port=5006, samples=8, timeout=0.3):
    """
    Estimate the offset needed to convert UAV local time into GSN local time.
    Returns (offset_seconds, best_rtt_seconds).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    measurements = []

    try:
        for seq in range(samples):
            t1 = time.time()
            payload = SYNC_PACKET.pack(SYNC_REQ, seq, t1)
            sock.sendto(payload, (server_ip, port))

            try:
                data, _ = sock.recvfrom(1024)
                t4 = time.time()
            except socket.timeout:
                continue

            if len(data) != SYNC_RESPONSE_PACKET.size:
                continue

            try:
                msg_type, resp_seq, echoed_t1, t2, t3 = SYNC_RESPONSE_PACKET.unpack(data)
            except struct.error:
                continue

            if msg_type != SYNC_RESP or resp_seq != seq:
                continue
            if abs(echoed_t1 - t1) > 0.050:
                continue

            rtt = (t4 - t1) - (t3 - t2)
            offset = ((t2 - t1) + (t3 - t4)) / 2.0
            measurements.append((rtt, offset))
            time.sleep(0.02)
    finally:
        sock.close()

    if not measurements:
        return 0.0, None

    measurements.sort(key=lambda item: item[0])
    best_rtt, best_offset = measurements[0]
    return best_offset, best_rtt
