import queue
import socket
import struct
import threading
import time

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from time_sync import ClockSyncServer

try:
    import av
except Exception:
    av = None


VIDEO_HEADER_PACKET = struct.Struct("!I I d I 1s ? ?")
VIDEO_PAYLOAD_PACKET = struct.Struct("!I I")
CODEC_H264 = b"4"
CODEC_JPEG = b"J"


class H264AccessUnitDecoder:
    def __init__(self):
        if av is None:
            raise RuntimeError(
                "PyAV (`av`) is required on the GSN side for H.264 video decoding."
            )
        self._codec = None
        self.reset()

    def reset(self):
        self._codec = av.CodecContext.create("h264", "r")
        self._codec.thread_count = 0

    def decode(self, data):
        packets = self._codec.parse(data)
        if not packets:
            packets = self._codec.parse(b"")

        frames = []
        for packet in packets:
            frames.extend(self._codec.decode(packet))
        return frames


class GSNReceiver:
    def __init__(
        self,
        get_aes_key,
        on_frame,
        video_port=5005,
        frame_timeout=0.4,
        sync_port=5006,
        decode_queue_size=32,
        on_eve_frame=None,
    ):
        """
        get_aes_key(epoch) -> bytes
        on_frame(frame_bgr, latency_ms)
        """
        self.get_aes_key = get_aes_key
        self.on_frame = on_frame
        self.on_eve_frame = on_eve_frame
        self.video_port = video_port
        self.frame_timeout = frame_timeout
        self.sync_port = sync_port
        self.frames = {}
        self.queue = queue.Queue(maxsize=decode_queue_size)
        self.last_completed_fid = -1
        self.last_header_fid = -1
        self.last_header_epoch = -1
        self.clock_sync_server = ClockSyncServer(port=self.sync_port)
        self.h264_decoder = H264AccessUnitDecoder() if av is not None else None
        self.need_h264_keyframe = threading.Event()
        self.warned_h264_missing = False

    def start(self):
        self.clock_sync_server.start()
        threading.Thread(target=self._recv, daemon=True).start()
        threading.Thread(target=self._show, daemon=True).start()

    def _drop_stale_frames(self, now):
        stale = [
            fid
            for fid, info in self.frames.items()
            if now - info["created_at"] > self.frame_timeout
        ]
        if stale:
            self.need_h264_keyframe.set()
        for fid in stale:
            self.frames.pop(fid, None)

    def _push_latest(self, item):
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.need_h264_keyframe.set()
            except queue.Empty:
                pass
        self.queue.put_nowait(item)

    def _reset_stream_state(self, reason):
        print(f"[GSNReceiver] stream reset detected: {reason}")
        self.frames.clear()
        self.last_completed_fid = -1
        self.last_header_fid = -1
        self.last_header_epoch = -1
        self.need_h264_keyframe.set()
        if self.h264_decoder is not None:
            self.h264_decoder.reset()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

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
                fid, epoch, ts, cnt, codec_tag, keyframe, encrypted = VIDEO_HEADER_PACKET.unpack(data[1:])

                # UAV reboot/new session: frame_id or epoch rolled back.
                if (
                    (self.last_header_epoch >= 0 and epoch < self.last_header_epoch)
                    or (self.last_header_fid >= 0 and fid + 30 < self.last_header_fid)
                ):
                    self._reset_stream_state(
                        f"epoch {self.last_header_epoch}->{epoch}, fid {self.last_header_fid}->{fid}"
                    )

                self.last_header_epoch = epoch
                self.last_header_fid = fid

                if fid <= self.last_completed_fid:
                    continue
                self.frames[fid] = {
                    "epoch": epoch,
                    "ts": ts,
                    "codec": codec_tag,
                    "keyframe": keyframe,
                    "encrypted": encrypted,
                    "pkts": {},
                    "max": cnt,
                    "created_at": now,
                }

            elif t == b"P":
                fid, pid = VIDEO_PAYLOAD_PACKET.unpack(data[1:9])
                info = self.frames.get(fid)
                if info is None:
                    continue

                info["pkts"][pid] = data[9:]
                if len(info["pkts"]) == info["max"]:
                    try:
                        blob = b"".join(info["pkts"][i] for i in range(info["max"]))
                    except KeyError:
                        self.need_h264_keyframe.set()
                        self.frames.pop(fid, None)
                        continue

                    self._push_latest(
                        (
                            fid,
                            info["epoch"],
                            info["ts"],
                            info["codec"],
                            info["keyframe"],
                            info["encrypted"],
                            blob,
                        )
                    )
                    self.last_completed_fid = max(self.last_completed_fid, fid)
                    self.frames.pop(fid, None)

                    old_fids = [old_fid for old_fid in self.frames if old_fid < fid]
                    for old_fid in old_fids:
                        self.frames.pop(old_fid, None)

    def _decode_payload(self, codec_tag, keyframe, payload):
        if codec_tag == CODEC_JPEG:
            return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)

        if codec_tag != CODEC_H264:
            return None

        if self.h264_decoder is None:
            if not self.warned_h264_missing:
                print("[GSNReceiver] missing PyAV (`av`) dependency; cannot decode H.264 stream.")
                self.warned_h264_missing = True
            return None

        if self.need_h264_keyframe.is_set():
            if not keyframe:
                return None
            self.h264_decoder.reset()
            self.need_h264_keyframe.clear()

        try:
            frames = self.h264_decoder.decode(payload)
        except Exception as exc:
            self.need_h264_keyframe.set()
            print(f"[GSNReceiver] H.264 decode error, waiting for next keyframe: {exc}")
            return None

        if not frames:
            return None

        return frames[-1].to_ndarray(format="bgr24")

    @staticmethod
    def _noise_frame(reference=None):
        if reference is not None:
            h, w = reference.shape[:2]
        else:
            h, w = 360, 640
        return np.random.randint(0, 256, (max(1, h), max(1, w), 3), dtype=np.uint8)

    def _emit_eve_frame(self, frame, encrypted):
        if self.on_eve_frame is None:
            return
        try:
            self.on_eve_frame(frame, encrypted)
        except Exception as exc:
            print(f"[GSNReceiver] EVE frame callback error: {exc}")

    def _show(self):
        last_epoch = None
        aes_cipher = None

        while True:
            _, epoch, ts, codec_tag, keyframe, encrypted, blob = self.queue.get()

            if encrypted:
                key = self.get_aes_key(epoch)
                if key is None:
                    self._emit_eve_frame(self._noise_frame(), True)
                    continue

                if epoch != last_epoch or aes_cipher is None:
                    aes_cipher = AESGCM(key)
                    last_epoch = epoch
                    self.need_h264_keyframe.set()

                nonce, cipher = blob[:12], blob[12:]
                try:
                    payload = aes_cipher.decrypt(nonce, cipher, None)
                except Exception:
                    self._emit_eve_frame(self._noise_frame(), True)
                    continue
            else:
                payload = blob
                aes_cipher = None
                last_epoch = None

            frame = self._decode_payload(codec_tag, keyframe, payload)
            if frame is None:
                continue

            if encrypted:
                self._emit_eve_frame(self._noise_frame(frame), True)
            else:
                self._emit_eve_frame(frame.copy(), False)

            # Guard against transient future timestamps while clock sync is warming up.
            latency = max(0.0, (time.time() - ts) * 1000)
            self.on_frame(frame, latency)
