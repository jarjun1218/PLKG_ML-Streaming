import io
import os
import queue
import socket
import struct
import threading
import time

import cv2
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from picamera2 import Picamera2
from time_sync import estimate_clock_offset

try:
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
except Exception:
    MJPEGEncoder = None
    FileOutput = None

cv2.setNumThreads(1)


class LatestEncodedFrameBuffer(io.BufferedIOBase):
    """Keep only the newest encoded frame from Picamera2's encoder output."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0

    def write(self, data):
        if not data:
            return 0
        with self._lock:
            self._seq += 1
            self._latest = (self._seq, time.time(), bytes(data))
        return len(data)

    def writable(self):
        return True

    def get_latest(self, last_seq):
        with self._lock:
            if self._latest is None:
                return None
            if self._latest[0] == last_seq:
                return None
            return self._latest

    def flush(self):
        return None


class UAVVideoStreamer:
    def __init__(
        self,
        get_video_key,
        gsn_ip,
        port=5005,
        chunk=4096,
        preview=False,
        debug=False,
        window_name="UAV Camera",
        preview_wait=1,
        resolution=(640, 360),
        fps=30,
        jpeg_quality=40,
        flip_code=None,
        fixed_exposure_us=12000,
        analogue_gain=1.0,
        use_hardware_mjpeg=True,
        sync_port=5006,
        sync_samples=8,
        sync_retry_interval=1.0,
        sync_refresh_interval=15.0,
        initial_sync_timeout=3.0,
    ):
        """
        get_video_key() -> (epoch, aes_key)
        """
        self.get_video_key = get_video_key
        self.gsn_ip = gsn_ip
        self.port = port
        self.chunk = chunk
        self.resolution = resolution
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.flip_code = flip_code
        self.fixed_exposure_us = fixed_exposure_us
        self.analogue_gain = analogue_gain
        self.use_hardware_mjpeg = use_hardware_mjpeg
        self.sync_port = sync_port
        self.sync_samples = sync_samples
        self.sync_retry_interval = sync_retry_interval
        self.sync_refresh_interval = sync_refresh_interval
        self.initial_sync_timeout = initial_sync_timeout

        self.preview = preview
        self.debug = debug
        self.window_name = window_name
        self.preview_wait = preview_wait

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.clock_offset = 0.0
        self.clock_offset_lock = threading.Lock()
        self.last_sync_rtt = None
        self.clock_sync_ready = threading.Event()
        self.clock_sync_stop = threading.Event()
        self.clock_sync_thread = None
        self.stats_lock = threading.Lock()
        self.stats_stop = threading.Event()
        self.stats_thread = None
        self.stats = self._make_stats_bucket()

    @staticmethod
    def _make_stats_bucket():
        return {
            "window_start": time.time(),
            "frames_captured": 0,
            "frames_available": 0,
            "frames_sent": 0,
            "packets_sent": 0,
            "wire_bytes_sent": 0,
            "jpeg_bytes_sent": 0,
            "chunks_sent": 0,
            "queue_drops": 0,
            "encoder_skips": 0,
            "key_wait_frames": 0,
            "capture_time_s": 0.0,
            "jpeg_encode_time_s": 0.0,
            "encrypt_time_s": 0.0,
            "send_time_s": 0.0,
            "pre_send_queue_time_s": 0.0,
        }

    def _reset_stats(self):
        with self.stats_lock:
            self.stats = self._make_stats_bucket()

    def _record_stats(
        self,
        *,
        frames_captured=0,
        frames_available=0,
        frames_sent=0,
        packets_sent=0,
        wire_bytes_sent=0,
        jpeg_bytes_sent=0,
        chunks_sent=0,
        queue_drops=0,
        encoder_skips=0,
        key_wait_frames=0,
        capture_time_s=0.0,
        jpeg_encode_time_s=0.0,
        encrypt_time_s=0.0,
        send_time_s=0.0,
        pre_send_queue_time_s=0.0,
    ):
        with self.stats_lock:
            self.stats["frames_captured"] += frames_captured
            self.stats["frames_available"] += frames_available
            self.stats["frames_sent"] += frames_sent
            self.stats["packets_sent"] += packets_sent
            self.stats["wire_bytes_sent"] += wire_bytes_sent
            self.stats["jpeg_bytes_sent"] += jpeg_bytes_sent
            self.stats["chunks_sent"] += chunks_sent
            self.stats["queue_drops"] += queue_drops
            self.stats["encoder_skips"] += encoder_skips
            self.stats["key_wait_frames"] += key_wait_frames
            self.stats["capture_time_s"] += capture_time_s
            self.stats["jpeg_encode_time_s"] += jpeg_encode_time_s
            self.stats["encrypt_time_s"] += encrypt_time_s
            self.stats["send_time_s"] += send_time_s
            self.stats["pre_send_queue_time_s"] += pre_send_queue_time_s

    def _stats_loop(self):
        while not self.stats_stop.wait(1.0):
            with self.stats_lock:
                now = time.time()
                window_start = self.stats["window_start"]
                elapsed = max(now - window_start, 1e-6)
                frames_captured = self.stats["frames_captured"]
                frames_available = self.stats["frames_available"]
                frames_sent = self.stats["frames_sent"]
                packets_sent = self.stats["packets_sent"]
                wire_bytes_sent = self.stats["wire_bytes_sent"]
                jpeg_bytes_sent = self.stats["jpeg_bytes_sent"]
                chunks_sent = self.stats["chunks_sent"]
                queue_drops = self.stats["queue_drops"]
                encoder_skips = self.stats["encoder_skips"]
                key_wait_frames = self.stats["key_wait_frames"]
                capture_time_s = self.stats["capture_time_s"]
                jpeg_encode_time_s = self.stats["jpeg_encode_time_s"]
                encrypt_time_s = self.stats["encrypt_time_s"]
                send_time_s = self.stats["send_time_s"]
                pre_send_queue_time_s = self.stats["pre_send_queue_time_s"]
                self.stats = self._make_stats_bucket()

            fps_in = frames_available / elapsed
            avg_jpeg_kb = (jpeg_bytes_sent / frames_sent / 1024.0) if frames_sent else 0.0
            avg_chunks = (chunks_sent / frames_sent) if frames_sent else 0.0
            mbps = (wire_bytes_sent * 8.0) / elapsed / 1_000_000.0
            clock_offset_ms = self._get_clock_offset() * 1000.0
            capture_ms = (capture_time_s * 1000.0 / frames_captured) if frames_captured else 0.0
            jpeg_ms = (jpeg_encode_time_s * 1000.0 / frames_available) if frames_available else 0.0
            encrypt_ms = (encrypt_time_s * 1000.0 / frames_sent) if frames_sent else 0.0
            send_ms = (send_time_s * 1000.0 / frames_sent) if frames_sent else 0.0
            queue_ms = (pre_send_queue_time_s * 1000.0 / frames_sent) if frames_sent else 0.0

            print(
                "[UAV Stats] "
                f"fps_in={fps_in:.1f} "
                f"fps_out={frames_sent / elapsed:.1f} "
                f"pkts/s={packets_sent / elapsed:.1f} "
                f"mbps={mbps:.2f} "
                f"avg_jpeg_kb={avg_jpeg_kb:.1f} "
                f"avg_chunks={avg_chunks:.1f} "
                f"cap_ms={capture_ms:.1f} "
                f"jpg_ms={jpeg_ms:.1f} "
                f"enc_ms={encrypt_ms:.1f} "
                f"send_ms={send_ms:.1f} "
                f"queue_ms={queue_ms:.1f} "
                f"qdrop/s={queue_drops / elapsed:.1f} "
                f"encskip/s={encoder_skips / elapsed:.1f} "
                f"keywait/s={key_wait_frames / elapsed:.1f} "
                f"clock_ms={clock_offset_ms:.1f}"
            )

    def _get_clock_offset(self):
        with self.clock_offset_lock:
            return self.clock_offset

    def _update_clock_offset(self, offset, best_rtt):
        with self.clock_offset_lock:
            prev_offset = self.clock_offset
            had_sync = self.clock_sync_ready.is_set()
            self.clock_offset = offset
            self.last_sync_rtt = best_rtt
        self.clock_sync_ready.set()

        if not had_sync:
            print(
                f"[UAV] clock sync ready: offset={offset * 1000:.1f} ms, "
                f"best_rtt={best_rtt * 1000:.1f} ms"
            )
            return

        if abs(offset - prev_offset) >= 0.005:
            print(
                f"[UAV] clock sync updated: offset={offset * 1000:.1f} ms, "
                f"best_rtt={best_rtt * 1000:.1f} ms"
            )

    def _clock_sync_loop(self):
        waiting_logged = False

        while not self.clock_sync_stop.is_set():
            offset, best_rtt = estimate_clock_offset(
                self.gsn_ip,
                port=self.sync_port,
                samples=self.sync_samples,
            )

            if best_rtt is None:
                if not self.clock_sync_ready.is_set() and not waiting_logged:
                    print("[UAV] waiting for GSN clock sync server...")
                    waiting_logged = True
                wait_time = self.sync_retry_interval
            else:
                self._update_clock_offset(offset, best_rtt)
                waiting_logged = False
                wait_time = self.sync_refresh_interval

            if self.clock_sync_stop.wait(wait_time):
                break

    def _apply_camera_controls(self, cam):
        try:
            frame_duration_us = int(1_000_000 / max(self.fps, 1))
            controls = {
                "FrameRate": self.fps,
                "FrameDurationLimits": (frame_duration_us, frame_duration_us),
            }
            if self.fixed_exposure_us is not None:
                controls["AeEnable"] = False
                controls["ExposureTime"] = min(self.fixed_exposure_us, frame_duration_us)
                if self.analogue_gain is not None:
                    controls["AnalogueGain"] = self.analogue_gain
            cam.set_controls(controls)
        except Exception as e:
            print(f"[Cam Warning] {e}")

    def _send_jpeg_bytes(self, sock, frame_id, ts, jpg_bytes, aes_cipher, epoch):
        corrected_ts = ts + self._get_clock_offset()
        nonce = os.urandom(12)
        encrypt_start = time.perf_counter()
        encrypted = nonce + aes_cipher.encrypt(nonce, jpg_bytes, None)
        encrypt_time_s = time.perf_counter() - encrypt_start
        chunks = [
            encrypted[i : i + self.chunk]
            for i in range(0, len(encrypted), self.chunk)
        ]

        header = b"H" + struct.pack("!I I d I 1s", frame_id, epoch, corrected_ts, len(chunks), b"A")
        send_start = time.perf_counter()
        sock.sendto(header, (self.gsn_ip, self.port))

        for idx, ch in enumerate(chunks):
            pkt = b"P" + struct.pack("!I I", frame_id, idx) + ch
            sock.sendto(pkt, (self.gsn_ip, self.port))
        send_time_s = time.perf_counter() - send_start
        pre_send_queue_time_s = max(0.0, time.time() - ts)

        self._record_stats(
            frames_sent=1,
            packets_sent=1 + len(chunks),
            wire_bytes_sent=len(header) + sum(9 + len(ch) for ch in chunks),
            jpeg_bytes_sent=len(jpg_bytes),
            chunks_sent=len(chunks),
            encrypt_time_s=encrypt_time_s,
            send_time_s=send_time_s,
            pre_send_queue_time_s=pre_send_queue_time_s,
        )

    def _producer_software(self, cam):
        frame_id = 0
        try:
            while self.running:
                frame_id += 1

                try:
                    capture_start = time.perf_counter()
                    capture_ts = time.time()
                    frame = cam.capture_array()
                    capture_time_s = time.perf_counter() - capture_start
                    if self.flip_code is not None:
                        frame = cv2.flip(frame, self.flip_code)
                    self._record_stats(
                        frames_captured=1,
                        frames_available=1,
                        capture_time_s=capture_time_s,
                    )
                except Exception as e:
                    print(f"Capture error: {e}")
                    continue

                if self.preview:
                    preview_frame = cv2.resize(frame, (640, 480))
                    cv2.imshow(self.window_name, preview_frame)
                    key = cv2.waitKey(self.preview_wait) & 0xFF
                    if key == ord("q"):
                        print("[UAV] preview quit (q pressed)")
                        self.running = False
                        break

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self._record_stats(queue_drops=1)
                    except queue.Empty:
                        pass

                self.frame_queue.put((frame_id, capture_ts, frame))

        except Exception as e:
            print(f"[Producer Error] {e}")
            self.running = False

    def _consumer_software(self, sock):
        last_epoch = None
        aes_cipher = None

        try:
            while self.running:
                try:
                    frame_id, capture_ts, frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                while True:
                    try:
                        frame_id, capture_ts, frame = self.frame_queue.get_nowait()
                    except queue.Empty:
                        break

                encode_start = time.perf_counter()
                ok, jpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        self.jpeg_quality,
                        cv2.IMWRITE_JPEG_OPTIMIZE,
                        0,
                        cv2.IMWRITE_JPEG_PROGRESSIVE,
                        0,
                    ],
                )
                encode_time_s = time.perf_counter() - encode_start
                if not ok:
                    continue
                self._record_stats(jpeg_encode_time_s=encode_time_s)

                epoch, aes_key = self.get_video_key()
                if aes_key is None:
                    self._record_stats(key_wait_frames=1)
                    continue

                if epoch != last_epoch or aes_cipher is None:
                    aes_cipher = AESGCM(aes_key)
                    last_epoch = epoch

                self._send_jpeg_bytes(sock, frame_id, capture_ts, jpg.tobytes(), aes_cipher, epoch)

        except Exception as e:
            print(f"[Consumer Error] {e}")
            self.running = False

    def _consumer_hardware(self, sock, encoded_output):
        frame_id = 0
        last_seq = 0
        last_epoch = None
        aes_cipher = None

        try:
            while self.running:
                latest = encoded_output.get_latest(last_seq)
                if latest is None:
                    time.sleep(0.001)
                    continue

                seq, encoded_ts, jpg_bytes = latest
                self._record_stats(frames_available=1)
                if last_seq and seq > last_seq + 1:
                    self._record_stats(encoder_skips=seq - last_seq - 1)
                last_seq = seq
                frame_id += 1

                epoch, aes_key = self.get_video_key()
                if aes_key is None:
                    self._record_stats(key_wait_frames=1)
                    continue

                if epoch != last_epoch or aes_cipher is None:
                    aes_cipher = AESGCM(aes_key)
                    last_epoch = epoch

                self._send_jpeg_bytes(sock, frame_id, encoded_ts, jpg_bytes, aes_cipher, epoch)

        except Exception as e:
            print(f"[HW Consumer Error] {e}")
            self.running = False

    def _preview_loop_hardware(self, cam):
        while self.running and self.preview:
            try:
                preview_frame = cam.capture_array("lores")
                if self.flip_code is not None:
                    preview_frame = cv2.flip(preview_frame, self.flip_code)
                preview_frame = cv2.resize(preview_frame, (640, 480))
                cv2.imshow(self.window_name, preview_frame)
                key = cv2.waitKey(self.preview_wait) & 0xFF
                if key == ord("q"):
                    print("[UAV] preview quit (q pressed)")
                    self.running = False
                    break
            except Exception as e:
                print(f"[HW Preview Error] {e}")
                time.sleep(0.05)

    def _run_hardware(self, sock):
        if not self.use_hardware_mjpeg or MJPEGEncoder is None or FileOutput is None:
            return False

        cam = None
        preview_thread = None
        consumer_thread = None

        try:
            cam = Picamera2()
            config_kwargs = {
                "main": {"size": self.resolution, "format": "YUV420"},
                "buffer_count": 4,
                "queue": False,
            }
            if self.preview:
                config_kwargs["lores"] = {"size": self.resolution, "format": "RGB888"}
                config_kwargs["display"] = "lores"

            cam.configure(cam.create_video_configuration(**config_kwargs))
            self._apply_camera_controls(cam)

            encoded_output = LatestEncodedFrameBuffer()
            encoder = MJPEGEncoder()

            cam.start_encoder(encoder, FileOutput(encoded_output))
            cam.start()

            self.running = True
            consumer_thread = threading.Thread(
                target=self._consumer_hardware,
                args=(sock, encoded_output),
            )
            consumer_thread.start()

            if self.preview:
                preview_thread = threading.Thread(target=self._preview_loop_hardware, args=(cam,))
                preview_thread.start()

            print(
                f"[UAV] hardware MJPEG stream started: {self.resolution} @ {self.fps}fps, "
                f"chunk={self.chunk}, fixed_exposure={self.fixed_exposure_us}, "
                f"analogue_gain={self.analogue_gain}"
            )

            while self.running:
                time.sleep(0.5)

            return True

        except Exception as e:
            print(f"[UAV] hardware MJPEG unavailable, falling back to software JPEG: {e}")
            self.running = False
            try:
                if cam is not None:
                    cam.stop_encoder()
            except Exception:
                pass
            try:
                if cam is not None:
                    cam.stop()
            except Exception:
                pass
            try:
                if cam is not None:
                    cam.close()
            except Exception:
                pass
            return False
        finally:
            self.running = False
            if preview_thread is not None:
                preview_thread.join(timeout=2)
            if consumer_thread is not None:
                consumer_thread.join(timeout=2)
            try:
                if cam is not None:
                    cam.stop_encoder()
            except Exception:
                pass
            try:
                if cam is not None:
                    cam.stop()
            except Exception:
                pass
            try:
                if cam is not None:
                    cam.close()
            except Exception:
                pass

    def _run_software(self, sock):
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": self.resolution, "format": "RGB888"},
            buffer_count=2,
            queue=False,
        )
        cam.configure(config)
        cam.start()
        self._apply_camera_controls(cam)

        print(
            f"[UAV] software JPEG stream started: {self.resolution} @ {self.fps}fps, "
            f"JPEG={self.jpeg_quality}, chunk={self.chunk}, "
            f"fixed_exposure={self.fixed_exposure_us}, analogue_gain={self.analogue_gain}"
        )
        self.running = True

        producer_thread = threading.Thread(target=self._producer_software, args=(cam,))
        consumer_thread = threading.Thread(target=self._consumer_software, args=(sock,))
        producer_thread.start()
        consumer_thread.start()

        try:
            while self.running:
                time.sleep(0.5)
        finally:
            self.running = False
            producer_thread.join(timeout=2)
            consumer_thread.join(timeout=2)
            try:
                cam.stop()
            except Exception:
                pass
            try:
                cam.close()
            except Exception:
                pass

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        except Exception:
            pass
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)
        except Exception:
            pass
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)
        except Exception:
            pass

        try:
            self.clock_sync_stop.clear()
            self.clock_sync_ready.clear()
            self.stats_stop.clear()
            self._reset_stats()
            with self.clock_offset_lock:
                self.clock_offset = 0.0
                self.last_sync_rtt = None
            self.clock_sync_thread = threading.Thread(
                target=self._clock_sync_loop,
                daemon=True,
            )
            self.clock_sync_thread.start()
            self.stats_thread = threading.Thread(
                target=self._stats_loop,
                daemon=True,
            )
            self.stats_thread.start()

            if not self.clock_sync_ready.wait(timeout=self.initial_sync_timeout):
                print(
                    "[UAV] clock sync not ready before stream start; "
                    "streaming will continue and retry in background"
                )

            started_hw = self._run_hardware(sock)
            if not started_hw:
                self._run_software(sock)
        except KeyboardInterrupt:
            print("[UAV] Interrupted by user")
            self.running = False
        finally:
            self.running = False
            self.clock_sync_stop.set()
            self.stats_stop.set()
            if self.clock_sync_thread is not None:
                self.clock_sync_thread.join(timeout=2)
                self.clock_sync_thread = None
            if self.stats_thread is not None:
                self.stats_thread.join(timeout=2)
                self.stats_thread = None
            try:
                sock.close()
            except Exception:
                pass
            if self.preview:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception:
                    pass
