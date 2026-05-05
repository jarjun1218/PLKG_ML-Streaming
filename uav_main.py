#!/usr/bin/env python3
"""
uav_main.py (FINAL with CNN-Q)

Pipeline:
1. CSI -> CNN (cnn_basic) -> raw key (102-bit)
2. raw key -> CNN-Q -> key_uav'
3. key_uav' -> BCH syndrome helper -> send (epoch, CSI serial, helper)
4. GSN corrects its local quantized key using the BCH helper
5. key_uav'/corrected key -> SHA256 -> AES key (key_uav'')
6. AES key -> HMAC key confirmation
7. AES-GCM(key_uav'') encrypt video -> send (epoch)
"""

import threading
import time
import numpy as np
import torch
import socket

import sha256
from greycode_quantization import quantization_1
from data_collecting_processing.collect import CSISerialStreamer
from key_confirm import make_key_confirm

from models.cnn_basic import cnn_basic
torch.serialization.add_safe_globals([cnn_basic])

from bch_reconciliation import bch_encode_syndrome_b64, force_102_bits

from uav_sender import UAVKeySender
from uav_stream import UAVVideoStreamer


# ======================================================
# Config
# ======================================================
GSN_IP = "192.168.0.154"
CSI_PORT = "/dev/ttyUSB0"
CSI_BAUD = 115200
DEBUG = False
PREVIEW = False
VIDEO_RESOLUTION = (960, 540)
VIDEO_FPS = 15
VIDEO_H264_BITRATE = 8_000_000
VIDEO_H264_IPERIOD = 15
VIDEO_JPEG_QUALITY = 10
VIDEO_CHUNK = 4096
VIDEO_FLIP_CODE = None
VIDEO_FIXED_EXPOSURE_US = 20000
VIDEO_ANALOGUE_GAIN = 4
VIDEO_USE_HARDWARE_H264 = True
TIME_SYNC_PORT = 5006
TIME_SYNC_SAMPLES = 8
UAV_CONTROL_PORT = 5008
KEY_UPDATE_INTERVAL_SEC = 5.0

MODEL_CSI_PATH = "model_reserved/cnn_basic/model_final.pth"
MODEL_KEY_QUAN_PATH = "model_reserved/cnn_basic_quan/model_final.pth"
NO_USED_CARRIERS_CH1 = [0,1,2,3,4,5,11,32,59,60,61,62,63]
NO_USED_CARRIERS_CH13 = [0, 1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]


# ======================================================
# KeyState (memory-only, thread-safe)
# ======================================================
class KeyState:
    def __init__(self):
        self.lock = threading.Lock()
        self.epoch = -1
        self.serial = None         # CSI serial used for this key
        self.helper = None         # BCH syndrome helper (base64)
        self.confirm = None        # HMAC confirmation tag (hex)
        self.aes_key = None        # AES key (bytes)

    def update(self, serial, helper, aes_key):
        with self.lock:
            self.epoch += 1
            self.serial = serial
            self.helper = helper
            self.confirm = make_key_confirm(aes_key, self.epoch, serial, helper)
            self.aes_key = aes_key

    def invalidate(self):
        with self.lock:
            self.epoch = -1
            self.serial = None
            self.helper = None
            self.confirm = None
            self.aes_key = None

    def for_reconciliation(self):
        with self.lock:
            return self.epoch, self.serial, self.helper, self.confirm

    def for_video(self):
        with self.lock:
            return self.epoch, self.aes_key


key_state = KeyState()
uav_csi_watcher = None
keygen_resync_event = threading.Event()


# ======================================================
# CSI Watcher (UAV)
# ======================================================
class CSISerialWatcher:
    def __init__(self, port, baudrate):
        self._lock = threading.Lock()
        self._latest = {}

        self.streamer = CSISerialStreamer(
            port,
            baudrate,
            endpoint_type="UAV",
            write_file=False,
            callback=self._handle
        )

    def start(self):
        self.streamer.start()

    def snapshot(self):
        with self._lock:
            return self._latest.copy()

    def force_reset(self):
        with self._lock:
            self._latest.clear()
        self.streamer.force_reset()

    def _handle(self, raw):
        sample = self._parse(raw)
        if sample:
            with self._lock:
                self._latest["UAV"] = sample

    def _parse(self, line):
        if "serial_num:" not in line:
            return None
        try:
            parts = [p for p in line.split(",") if p != ""]
            if parts[0] != "serial_num:":
                return None

            serial = int(parts[1])
            tail = [float(x) for x in parts[5:] if x]
            amps = np.array(tail[:64], dtype=np.float32)
            amps = np.delete(amps, NO_USED_CARRIERS_CH13)
            if len(amps) != 51:
                return None
            peak = np.max(amps)
            if peak <= 0:
                return None

            return {
                "serial": serial,
                "csi": amps / peak
            }
        except:
            return None


# ======================================================
# CNN-Q key reconstruction
# ======================================================
def reconstruct_key_cnnq(model_q, raw_bits: str, debug: bool = False) -> str:
    """
    raw_bits: 102-bit string
    return: reconstructed 102-bit string
    """
    arr = np.array([int(b) for b in raw_bits], dtype=np.float32)

    # GUI-style input: two consecutive keys
    key_pair = np.stack([arr, arr], axis=0)   # (2, 102)
    key_pair = key_pair.reshape(1, 2, 102)    # (1, 2, 102)

    with torch.no_grad():
        out = model_q(torch.from_numpy(key_pair))
    if debug:
        print("[DEBUG CNN-Q] out type :", type(out))
        print("[DEBUG CNN-Q] out shape:", out.shape if hasattr(out, "shape") else "no shape")
        print("[DEBUG CNN-Q] out value:", out)
    out_np = out.detach().cpu().numpy()

    # 不管 shape 是 (1,102)、(102,) 還是怪的，全部攤平
    out_flat = out_np.reshape(-1)

    if out_flat.size != 102:
        raise ValueError(
            f"CNN-Q output size mismatch: expected 102, got {out_flat.size}"
        )

    bits = (out_flat > 0.5).astype(np.int32)

    return "".join(str(int(b)) for b in bits)

# ======================================================
# THREAD 1: Key generation + CNN-Q + BCH
# ======================================================
def keygen_thread():
    global uav_csi_watcher
    # --- Load models ---
    model_csi = torch.load(MODEL_CSI_PATH, map_location="cpu", weights_only=False)
    model_csi.eval()

    model_q = torch.load(MODEL_KEY_QUAN_PATH, map_location="cpu", weights_only=False)
    model_q.eval()

    watcher = CSISerialWatcher(CSI_PORT, CSI_BAUD)
    uav_csi_watcher = watcher
    watcher.start()

    last_serial = None
    next_keygen_time = 0.0
    print("[UAV] keygen (with CNN-Q) started")

    while True:
        if keygen_resync_event.is_set():
            keygen_resync_event.clear()
            last_serial = None
            next_keygen_time = 0.0
            print("[UAV] keygen resync requested; waiting for fresh CSI")

        s = watcher.snapshot().get("UAV")
        if not s:
            time.sleep(0.05)
            continue

        if s["serial"] == last_serial:
            continue
        last_serial = s["serial"]

        now = time.monotonic()
        if now < next_keygen_time:
            continue

        # === Step 1: CSI → CNN → raw key ===
        csi = s["csi"]
        c1 = csi.reshape(1, 1, 1, 51)
        c2 = np.zeros_like(c1)
        arr = np.concatenate([c1, c2], axis=2)
        arr = np.concatenate([arr, arr], axis=0)

        with torch.no_grad():
            out = model_csi(torch.from_numpy(arr).float())

        features = out[0].cpu().numpy().ravel()
        bits = quantization_1(features, Nbits=2, inbits=13, guard=0)

        raw_key = force_102_bits("".join(str(b) for b in bits))

        # === Step 2: CNN-Q reconstruction ===
        key_uav_p = reconstruct_key_cnnq(model_q, raw_key, debug=DEBUG)

        # === Step 3: BCH syndrome helper for reconciliation ===
        helper = bch_encode_syndrome_b64(key_uav_p)

        # === Step 5: privacy amplification / AES-256 key ===
        aes_key = sha256.sha_byte(key_uav_p)

        # === Step 6: key confirmation tag ===
        key_state.update(s["serial"], helper, aes_key)
        next_keygen_time = time.monotonic() + KEY_UPDATE_INTERVAL_SEC
        print(
            f"[UAV] new key epoch={key_state.epoch} "
            f"(next update in {KEY_UPDATE_INTERVAL_SEC:.1f}s)"
        )


def control_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UAV_CONTROL_PORT))
    print(f"[UAV] control channel listening on UDP/{UAV_CONTROL_PORT}")

    while True:
        data, addr = sock.recvfrom(1024)
        cmd = data.decode(errors="ignore").strip()
        if cmd != "RESET_CSI":
            continue

        print(f"[UAV] received RESET_CSI from {addr[0]}")
        key_state.invalidate()
        keygen_resync_event.set()
        watcher = uav_csi_watcher
        if watcher is None:
            print("[UAV] invalidated active key; CSI watcher is not ready yet")
            continue

        try:
            watcher.force_reset()
            print("[UAV] invalidated active key after RESET_CSI")
        except Exception as exc:
            print(f"[UAV] failed to reset CSI streamer: {exc}")


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    # Key generation thread
    threading.Thread(target=keygen_thread, daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()

    # Reconciliation helper sender
    sender = UAVKeySender(key_state.for_reconciliation, GSN_IP, debug=DEBUG)
    threading.Thread(target=sender.run, daemon=True).start()

    # Video streamer (preview in main thread is more stable)
    streamer = UAVVideoStreamer(
        key_state.for_video,
        GSN_IP,
        preview=PREVIEW,
        debug=DEBUG,
        resolution=VIDEO_RESOLUTION,
        fps=VIDEO_FPS,
        h264_bitrate=VIDEO_H264_BITRATE,
        h264_iperiod=VIDEO_H264_IPERIOD,
        jpeg_quality=VIDEO_JPEG_QUALITY,
        chunk=VIDEO_CHUNK,
        flip_code=VIDEO_FLIP_CODE,
        fixed_exposure_us=VIDEO_FIXED_EXPOSURE_US,
        analogue_gain=VIDEO_ANALOGUE_GAIN,
        use_hardware_h264=VIDEO_USE_HARDWARE_H264,
        sync_port=TIME_SYNC_PORT,
        sync_samples=TIME_SYNC_SAMPLES,
    )

    print("[UAV] system ready (press q to quit preview)")
    streamer.run()
