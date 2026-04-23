#!/usr/bin/env python3
"""
uav_main.py (FINAL with CNN-Q)

Pipeline:
1. CSI → CNN (cnn_basic) → raw key (102-bit)
2. raw key → CNN-Q → key_uav'
3-1. key_uav' → SHA256 → AES key (key_uav'')
3-2. key_uav' → RS parity → send (epoch, key_uav', parity)
4. AES-GCM(key_uav'') encrypt video → send (epoch)
"""

import threading
import time
import numpy as np
import torch

import sha256
from greycode_quantization import quantization_1
from data_collecting_processing.collect import CSISerialStreamer

from models.cnn_basic import cnn_basic
torch.serialization.add_safe_globals([cnn_basic])

from rs_ecc import rs_encode_parity_b64, force_102_bits

from uav_sender import UAVKeySender
from uav_stream import UAVVideoStreamer


# ======================================================
# Config
# ======================================================
GSN_IP = "192.168.0.149"
CSI_PORT = "/dev/ttyUSB0"
CSI_BAUD = 115200
DEBUG = False
PREVIEW = False
VIDEO_RESOLUTION = (960, 540)
VIDEO_FPS = 20
VIDEO_JPEG_QUALITY = 45
VIDEO_CHUNK = 1200

MODEL_CSI_PATH = "model_reserved/cnn_basic/model_final.pth"
MODEL_KEY_QUAN_PATH = "model_reserved/cnn_basic_quan/model_final.pth"


# ======================================================
# KeyState (memory-only, thread-safe)
# ======================================================
class KeyState:
    def __init__(self):
        self.lock = threading.Lock()
        self.epoch = -1
        self.key_uav_p = None      # reconstructed key (bitstring)
        self.parity = None         # RS parity (base64)
        self.aes_key = None        # AES key (bytes)

    def update(self, key_uav_p, parity, aes_key):
        with self.lock:
            self.epoch += 1
            self.key_uav_p = key_uav_p
            self.parity = parity
            self.aes_key = aes_key

    def for_parity(self):
        with self.lock:
            return self.epoch, self.key_uav_p, self.parity

    def for_video(self):
        with self.lock:
            return self.epoch, self.aes_key


key_state = KeyState()


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

            start = next((i for i, v in enumerate(tail) if v > 3.0), None)
            if start is None or len(tail) < start + 64:
                return None

            amps = np.array(tail[start:start + 64], dtype=np.float32)
            amps = np.delete(amps, [0,1,2,3,4,5,11,32,59,60,61,62,63])
            if len(amps) != 51:
                return None

            return {
                "serial": serial,
                "csi": amps / 61.846
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
# THREAD 1: Key generation + CNN-Q + RS
# ======================================================
def keygen_thread():
    # --- Load models ---
    model_csi = torch.load(MODEL_CSI_PATH, map_location="cpu", weights_only=False)
    model_csi.eval()

    model_q = torch.load(MODEL_KEY_QUAN_PATH, map_location="cpu", weights_only=False)
    model_q.eval()

    watcher = CSISerialWatcher(CSI_PORT, CSI_BAUD)
    watcher.start()

    last_serial = None
    print("[UAV] keygen (with CNN-Q) started")

    while True:
        s = watcher.snapshot().get("UAV")
        if not s:
            time.sleep(0.05)
            continue

        if s["serial"] == last_serial:
            continue
        last_serial = s["serial"]

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

        # === Step 3-1: AES key ===
        aes_key = sha256.sha_byte(key_uav_p)

        # === Step 3-2: RS parity (based on reconstructed key) ===
        parity = rs_encode_parity_b64(key_uav_p)

        key_state.update(key_uav_p, parity, aes_key)
        print(f"[UAV] new key epoch={key_state.epoch}")


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    # Key generation thread
    threading.Thread(target=keygen_thread, daemon=True).start()

    # Parity sender
    sender = UAVKeySender(key_state.for_parity, GSN_IP, debug=DEBUG)
    threading.Thread(target=sender.run, daemon=True).start()

    # Video streamer (preview in main thread is more stable)
    streamer = UAVVideoStreamer(
        key_state.for_video,
        GSN_IP,
        preview=PREVIEW,
        debug=DEBUG,
        resolution=VIDEO_RESOLUTION,
        fps=VIDEO_FPS,
        jpeg_quality=VIDEO_JPEG_QUALITY,
        chunk=VIDEO_CHUNK,
    )

    print("[UAV] system ready (press q to quit preview)")
    streamer.run()
