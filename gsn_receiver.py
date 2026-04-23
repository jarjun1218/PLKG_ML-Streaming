import queue
import socket
import struct
import threading
import time

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class GSNReceiver:
    def __init__(self, get_aes_key, on_frame, video_port=5005, frame_timeout=0.4):
        """
        get_aes_key(epoch) -> bytes
        on_frame(frame_bgr, latency_ms)
        """
        self.get_aes_key = get_aes_key
        self.on_frame = on_frame
        self.video_port = video_port
        self.frame_timeout = frame_timeout
        self.frames = {}
        self.queue = queue.Queue(maxsize=1)
        self.last_completed_fid = -1

    def start(self):
        threading.Thread(target=self._recv, daemon=True).start()
        threading.Thread(target=self._show, daemon=True).start()

    def _drop_stale_frames(self, now):
        stale = [
            fid
            for fid, info in self.frames.items()
            if now - info["created_at"] > self.frame_timeout
        ]
        for fid in stale:
            self.frames.pop(fid, None)

    def _push_latest(self, item):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put_nowait(item)

    def _recv(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        except Exception:
            pass
        sock.bind(("0.0.0.0", self.video_port))

        while True:
            data, _ = sock.recvfrom(65535)
            now = time.time()
            self._drop_stale_frames(now)

            t = data[0:1]

            if t == b"H":
                fid, epoch, ts, cnt, _ = struct.unpack("!I I d I 1s", data[1:])
                if fid <= self.last_completed_fid:
                    continue
                self.frames[fid] = {
                    "epoch": epoch,
                    "ts": ts,
                    "pkts": {},
                    "max": cnt,
                    "created_at": now,
                }

            elif t == b"P":
                fid, pid = struct.unpack("!I I", data[1:9])
                info = self.frames.get(fid)
                if info is None:
                    continue

                info["pkts"][pid] = data[9:]
                if len(info["pkts"]) == info["max"]:
                    try:
                        blob = b"".join(info["pkts"][i] for i in range(info["max"]))
                    except KeyError:
                        self.frames.pop(fid, None)
                        continue

                    self._push_latest((fid, info["epoch"], info["ts"], blob))
                    self.last_completed_fid = max(self.last_completed_fid, fid)
                    self.frames.pop(fid, None)

                    old_fids = [old_fid for old_fid in self.frames if old_fid < fid]
                    for old_fid in old_fids:
                        self.frames.pop(old_fid, None)

    def _show(self):
        last_epoch = None
        aes_cipher = None

        while True:
            _, epoch, ts, blob = self.queue.get()
            key = self.get_aes_key(epoch)
            if key is None:
                continue

            if epoch != last_epoch or aes_cipher is None:
                aes_cipher = AESGCM(key)
                last_epoch = epoch

            nonce, cipher = blob[:12], blob[12:]
            try:
                jpg = aes_cipher.decrypt(nonce, cipher, None)
            except Exception:
                continue

            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            latency = (time.time() - ts) * 1000
            self.on_frame(frame, latency)
