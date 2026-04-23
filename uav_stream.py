import os
import queue
import socket
import struct
import threading
import time

import cv2
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from picamera2 import Picamera2


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

        self.preview = preview
        self.debug = debug
        self.window_name = window_name
        self.preview_wait = preview_wait

        # Keep only the newest frame to avoid queue-backed latency growth.
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False

    def _producer(self, cam):
        frame_id = 0
        try:
            while self.running:
                frame_id += 1

                try:
                    frame = cam.capture_array()
                    if self.flip_code is not None:
                        frame = cv2.flip(frame, self.flip_code)
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

                ok, jpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    continue

                jpg_bytes = jpg.tobytes()

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.frame_queue.put((frame_id, jpg_bytes))

        except Exception as e:
            print(f"[Producer Error] {e}")
            self.running = False

    def _consumer(self, sock):
        last_epoch = None
        aes_cipher = None

        try:
            while self.running:
                try:
                    frame_id, jpg_bytes = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                epoch, aes_key = self.get_video_key()
                if aes_key is None:
                    continue

                if epoch != last_epoch or aes_cipher is None:
                    aes_cipher = AESGCM(aes_key)
                    last_epoch = epoch

                nonce = os.urandom(12)
                encrypted = nonce + aes_cipher.encrypt(nonce, jpg_bytes, None)
                chunks = [
                    encrypted[i : i + self.chunk]
                    for i in range(0, len(encrypted), self.chunk)
                ]
                ts = time.time()

                header = (
                    b"H"
                    + struct.pack("!I I d I 1s", frame_id, epoch, ts, len(chunks), b"A")
                )
                sock.sendto(header, (self.gsn_ip, self.port))

                for idx, ch in enumerate(chunks):
                    pkt = b"P" + struct.pack("!I I", frame_id, idx) + ch
                    sock.sendto(pkt, (self.gsn_ip, self.port))

        except Exception as e:
            print(f"[Consumer Error] {e}")
            self.running = False

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

        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": self.resolution, "format": "RGB888"},
            buffer_count=2,
            queue=False,
        )
        cam.configure(config)
        cam.start()

        try:
            cam.set_controls({"FrameRate": self.fps, "ExposureTime": 20000})
        except Exception as e:
            print(f"[Cam Warning] {e}")

        print(
            f"[UAV] video stream started: {self.resolution} @ {self.fps}fps, "
            f"JPEG={self.jpeg_quality}, chunk={self.chunk}"
        )
        self.running = True

        producer_thread = threading.Thread(target=self._producer, args=(cam,))
        consumer_thread = threading.Thread(target=self._consumer, args=(sock,))
        producer_thread.start()
        consumer_thread.start()

        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("[UAV] Interrupted by user")
            self.running = False
        finally:
            self.running = False
            producer_thread.join(timeout=2)
            consumer_thread.join(timeout=2)

            try:
                cam.stop()
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
            if self.preview:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception:
                    pass
