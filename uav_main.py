#!/usr/bin/env python3
"""
uav_main.py

Default key agreement path:
1. CSI preprocessing -> wait for two consecutive UAV CSI samples
2. [raw key serial N, raw key serial N+1] -> CNN-Q -> pending UAV key
3. pending key -> BCH syndrome helper -> send (epoch, CSI serial pair, helper)
4. GSN waits for the same serial pair, then corrects its serial N raw key
5. GSN verifies HMAC key confirmation and sends KEY_ACK
6. UAV promotes pending key to active video key only after KEY_ACK
"""

import argparse
import threading
import time
import os
import numpy as np
import torch
import socket

import sha256
from greycode_quantization import quantization_1
from data_collecting_processing.collect import CSISerialStreamer
from fetch_ESP32_CSI import preprocess_csi_line
from key_confirm import make_key_confirm

from models.cnn_basic import cnn_basic
torch.serialization.add_safe_globals([cnn_basic])

from bch_reconciliation import bch_encode_syndrome_b64, force_102_bits

from uav_sender import UAVKeySender
from uav_stream import UAVVideoStreamer
try:
    from demo_telemetry import DEMO_TELEMETRY_PORT, DemoTelemetrySender, LiveCSITelemetrySender
    DEMO_TELEMETRY_IMPORT_ERROR = None
except ImportError as exc:
    DEMO_TELEMETRY_PORT = 5009
    DemoTelemetrySender = None
    LiveCSITelemetrySender = None
    DEMO_TELEMETRY_IMPORT_ERROR = exc


def _read_optional_number_env(name, default=None, cast=float):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    if not value or value.lower() in ("auto", "none", "null", "off"):
        return None
    return cast(value)


def _parse_resolution(value):
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        raise argparse.ArgumentTypeError("resolution must use WIDTHxHEIGHT, for example 1280x720")
    width_text, height_text = text.split("x", 1)
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution width and height must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution width and height must be positive")
    return width, height


def _parse_optional_int(value):
    text = str(value).strip().lower()
    if text in ("auto", "none", "null", "off"):
        return None
    return int(text)


def _parse_optional_float(value):
    text = str(value).strip().lower()
    if text in ("auto", "none", "null", "off"):
        return None
    return float(text)


# ======================================================
# Config
# ======================================================
DEFAULT_GSN_IP = "192.168.0.149"
CSI_PORT = "/dev/ttyUSB0"
CSI_BAUD = 115200
DEBUG = False
PREVIEW = False
VIDEO_RESOLUTION = (1280, 720)
VIDEO_FPS = 30
VIDEO_H264_BITRATE = 8_000_000
VIDEO_H264_IPERIOD = 15
VIDEO_JPEG_QUALITY = 10
VIDEO_CHUNK = 4096
VIDEO_FLIP_CODE = None
VIDEO_FIXED_EXPOSURE_US = _read_optional_number_env("PLKG_VIDEO_FIXED_EXPOSURE_US", None, int)
VIDEO_ANALOGUE_GAIN = _read_optional_number_env("PLKG_VIDEO_ANALOGUE_GAIN", None, float)
VIDEO_USE_HARDWARE_H264 = True
TIME_SYNC_PORT = 5006
TIME_SYNC_SAMPLES = 8
UAV_CONTROL_PORT = 5008
KEY_UPDATE_INTERVAL_SEC = 10.0
KEY_ACK_TIMEOUT_SEC = float(os.environ.get("PLKG_KEY_ACK_TIMEOUT_SEC", "12.0"))
LIVE_CSI_TELEMETRY_INTERVAL_SEC = 0.5
CSI_PAIR_WAIT_LOG_INTERVAL_SEC = 5.0
CSI_PAIR_WAIT_RESET_LOGS = 3
CSI_PAIR_WAIT_RESET_COOLDOWN_SEC = 20.0
DEMO_TELEMETRY_ENABLED = DemoTelemetrySender is not None
KEY_SOURCE = os.environ.get("PLKG_KEY_SOURCE", "cnnq").strip().lower()

MODEL_CSI_PATH = "model_reserved/cnn_basic/model_final_test.pth"
MODEL_KEY_QUAN_PATH = "model_reserved/cnn_basic_quan/model_final.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the UAV PLKG/video sender.")
    parser.add_argument(
        "--GSN-IP",
        "--gsn-ip",
        dest="gsn_ip",
        default=DEFAULT_GSN_IP,
        help=f"GSN receiver IP address. Default: {DEFAULT_GSN_IP}",
    )
    parser.add_argument(
        "--video-resolution",
        type=_parse_resolution,
        default=VIDEO_RESOLUTION,
        metavar="WIDTHxHEIGHT",
        help=f"Camera/video resolution. Default: {VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]}",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=VIDEO_FPS,
        help=f"Camera/video FPS. Default: {VIDEO_FPS}",
    )
    parser.add_argument(
        "--video-bitrate",
        type=int,
        default=VIDEO_H264_BITRATE,
        help=f"H.264 bitrate in bits/sec. Default: {VIDEO_H264_BITRATE}",
    )
    parser.add_argument(
        "--video-iperiod",
        type=_parse_optional_int,
        default=VIDEO_H264_IPERIOD,
        help=f"H.264 I-frame interval. Use 'auto' for FPS-based interval. Default: {VIDEO_H264_IPERIOD}",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=VIDEO_JPEG_QUALITY,
        help=f"Software JPEG quality fallback. Default: {VIDEO_JPEG_QUALITY}",
    )
    parser.add_argument(
        "--video-chunk",
        type=int,
        default=VIDEO_CHUNK,
        help=f"UDP payload chunk size. Default: {VIDEO_CHUNK}",
    )
    parser.add_argument(
        "--fixed-exposure-us",
        type=_parse_optional_int,
        default=VIDEO_FIXED_EXPOSURE_US,
        help="Fixed camera exposure time in microseconds. Use 'auto' for auto exposure. Default: auto",
    )
    parser.add_argument(
        "--analogue-gain",
        type=_parse_optional_float,
        default=VIDEO_ANALOGUE_GAIN,
        help="Fixed camera analogue gain. Use 'auto' for auto gain. Default: auto",
    )
    parser.add_argument(
        "--software-jpeg",
        action="store_true",
        help="Disable hardware H.264 and use software JPEG fallback.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show local camera preview on the UAV.",
    )
    return parser.parse_args()


# ======================================================
# KeyState (memory-only, thread-safe)
# ======================================================
class KeyState:
    def __init__(self):
        self.lock = threading.Lock()
        self.epoch = -1
        self.next_epoch = 0
        self.serial = None         # first CSI serial used for this key
        self.serial_pair = None    # two consecutive CSI serials used by CNN/CNN-Q
        self.serial_token = None   # encoded serial pair sent to GSN
        self.helper = None         # BCH syndrome helper (base64)
        self.confirm = None        # HMAC confirmation tag (hex)
        self.aes_key = None        # AES key (bytes)
        self.rssi = None           # UAV RSSI
        self.raw_csi = None        # first preprocessed UAV CSI
        self.raw_csi_2 = None      # second preprocessed UAV CSI
        self.raw_key = None        # direct quantization from first raw CSI
        self.raw_key_2 = None      # direct quantization from second raw CSI
        self.cnn_csi = None        # CNN-corrected CSI/features
        self.cnn_key = None        # quantization from CNN-corrected CSI
        self.cnnq_key = None       # CNN-Q correction from direct raw key
        self.corrected_key = None  # active key selected for BCH/AES
        self.live_cnn_csi = None
        self.live_cnn_serial_pair = None
        self.pending = None

    @staticmethod
    def _normalize_pair(serial_pair):
        if isinstance(serial_pair, (tuple, list)):
            return tuple(int(item) for item in serial_pair)
        return (int(serial_pair),)

    def _snapshot_from_args(
        self,
        serial_pair,
        helper,
        aes_key,
        rssi=None,
        raw_csi=None,
        raw_csi_2=None,
        raw_key=None,
        raw_key_2=None,
        cnn_csi=None,
        cnn_key=None,
        cnnq_key=None,
        corrected_key=None,
    ):
        pair = self._normalize_pair(serial_pair)
        serial_token = ",".join(str(item) for item in pair)
        return {
            "serial": pair[0],
            "serial_pair": pair,
            "serial_token": serial_token,
            "helper": helper,
            "aes_key": aes_key,
            "rssi": [rssi] if rssi is not None and not isinstance(rssi, (list, tuple)) else rssi,
            "raw_csi": None if raw_csi is None else np.asarray(raw_csi, dtype=np.float32).copy(),
            "raw_csi_2": None if raw_csi_2 is None else np.asarray(raw_csi_2, dtype=np.float32).copy(),
            "raw_key": raw_key,
            "raw_key_2": raw_key_2,
            "cnn_csi": None if cnn_csi is None else np.asarray(cnn_csi, dtype=np.float32).copy(),
            "cnn_key": cnn_key,
            "cnnq_key": cnnq_key,
            "corrected_key": corrected_key,
        }

    def stage_pending(
        self,
        serial_pair,
        helper,
        aes_key,
        rssi=None,
        raw_csi=None,
        raw_csi_2=None,
        raw_key=None,
        raw_key_2=None,
        cnn_csi=None,
        cnn_key=None,
        cnnq_key=None,
        corrected_key=None,
    ):
        snapshot = self._snapshot_from_args(
            serial_pair,
            helper,
            aes_key,
            rssi=rssi,
            raw_csi=raw_csi,
            raw_csi_2=raw_csi_2,
            raw_key=raw_key,
            raw_key_2=raw_key_2,
            cnn_csi=cnn_csi,
            cnn_key=cnn_key,
            cnnq_key=cnnq_key,
            corrected_key=corrected_key,
        )

        with self.lock:
            if self.pending is not None:
                return None
            pending_epoch = self.next_epoch
            self.next_epoch += 1
            snapshot["epoch"] = pending_epoch
            snapshot["confirm"] = make_key_confirm(
                snapshot["aes_key"],
                pending_epoch,
                snapshot["serial_token"],
                snapshot["helper"],
            )
            snapshot["created_at"] = time.monotonic()
            self.pending = snapshot
            self.live_cnn_csi = None if snapshot["cnn_csi"] is None else snapshot["cnn_csi"].copy()
            self.live_cnn_serial_pair = snapshot["serial_pair"]
            return pending_epoch

    def activate_pending_ack(self, epoch, serial_token, confirm):
        with self.lock:
            if self.pending is None:
                return False, "no pending key"
            pending = self.pending
            if int(epoch) != pending["epoch"]:
                return False, f"pending epoch mismatch ({pending['epoch']} != {epoch})"
            if str(serial_token) != pending["serial_token"]:
                return False, "pending serial token mismatch"
            if str(confirm) != pending["confirm"]:
                return False, "pending confirm tag mismatch"

            self.epoch = pending["epoch"]
            self.serial = pending["serial"]
            self.serial_pair = pending["serial_pair"]
            self.serial_token = pending["serial_token"]
            self.helper = pending["helper"]
            self.confirm = pending["confirm"]
            self.aes_key = pending["aes_key"]
            self.rssi = pending["rssi"]
            self.raw_csi = None if pending["raw_csi"] is None else pending["raw_csi"].copy()
            self.raw_csi_2 = None if pending["raw_csi_2"] is None else pending["raw_csi_2"].copy()
            self.raw_key = pending["raw_key"]
            self.raw_key_2 = pending["raw_key_2"]
            self.cnn_csi = None if pending["cnn_csi"] is None else pending["cnn_csi"].copy()
            self.cnn_key = pending["cnn_key"]
            self.cnnq_key = pending["cnnq_key"]
            self.corrected_key = pending["corrected_key"]
            self.pending = None
            return True, f"activated epoch {self.epoch}"

    def expire_pending(self, timeout_sec):
        with self.lock:
            if self.pending is None:
                return None
            age = time.monotonic() - self.pending["created_at"]
            if age < timeout_sec:
                return None
            expired = self.pending
            self.pending = None
            return expired

    def has_pending(self):
        with self.lock:
            return self.pending is not None

    def invalidate(self):
        with self.lock:
            self.epoch = -1
            self.next_epoch = 0
            self.serial = None
            self.serial_pair = None
            self.serial_token = None
            self.helper = None
            self.confirm = None
            self.aes_key = None
            self.rssi = None
            self.raw_csi = None
            self.raw_csi_2 = None
            self.raw_key = None
            self.raw_key_2 = None
            self.cnn_csi = None
            self.cnn_key = None
            self.cnnq_key = None
            self.corrected_key = None
            self.live_cnn_csi = None
            self.live_cnn_serial_pair = None
            self.pending = None

    def for_reconciliation(self):
        with self.lock:
            if self.pending is None:
                return -1, None, None, None
            return (
                self.pending["epoch"],
                self.pending["serial_token"],
                self.pending["helper"],
                self.pending["confirm"],
            )

    def for_video(self):
        with self.lock:
            return self.epoch, self.aes_key

    def current_epoch(self):
        with self.lock:
            return self.epoch

    def current_cnn_csi(self):
        with self.lock:
            source_csi = self.live_cnn_csi if self.live_cnn_csi is not None else self.cnn_csi
            source_pair = self.live_cnn_serial_pair if self.live_cnn_serial_pair is not None else self.serial_pair
            cnn_csi = None if source_csi is None else source_csi.copy()
            serial_pair = None if source_pair is None else tuple(source_pair)
            return cnn_csi, serial_pair

    def update_live_cnn_csi(self, serial_pair, cnn_csi):
        if cnn_csi is None:
            return
        pair = tuple(int(item) for item in serial_pair)
        with self.lock:
            self.live_cnn_csi = np.asarray(cnn_csi, dtype=np.float32).copy()
            self.live_cnn_serial_pair = pair

    def for_demo_telemetry(self):
        with self.lock:
            source = self.pending
            if source is None:
                raw_csi = None if self.raw_csi is None else self.raw_csi.copy()
                raw_csi_2 = None if self.raw_csi_2 is None else self.raw_csi_2.copy()
                cnn_csi = None if self.cnn_csi is None else self.cnn_csi.copy()
                return (
                    self.epoch,
                    self.serial,
                    self.rssi,
                    raw_csi,
                    self.raw_key,
                    self.corrected_key,
                    cnn_csi,
                    self.cnn_key,
                    self.cnnq_key,
                    self.serial_pair,
                    raw_csi_2,
                    self.raw_key_2,
                )

            raw_csi = None if source["raw_csi"] is None else source["raw_csi"].copy()
            raw_csi_2 = None if source["raw_csi_2"] is None else source["raw_csi_2"].copy()
            cnn_csi = None if source["cnn_csi"] is None else source["cnn_csi"].copy()
            return (
                source["epoch"],
                source["serial"],
                source["rssi"],
                raw_csi,
                source["raw_key"],
                source["corrected_key"],
                cnn_csi,
                source["cnn_key"],
                source["cnnq_key"],
                source["serial_pair"],
                raw_csi_2,
                source["raw_key_2"],
            )


key_state = KeyState()
uav_csi_watcher = None
keygen_resync_event = threading.Event()


class VideoEncryptionState:
    def __init__(self, enabled=True):
        self._lock = threading.Lock()
        self._enabled = bool(enabled)

    def set_enabled(self, enabled):
        with self._lock:
            changed = self._enabled != bool(enabled)
            self._enabled = bool(enabled)
            return changed

    def is_enabled(self):
        with self._lock:
            return self._enabled


video_encryption_state = VideoEncryptionState(enabled=True)


def latest_uav_csi_for_telemetry():
    watcher = uav_csi_watcher
    if watcher is None:
        return None
    snap = watcher.snapshot().get("UAV")
    if not snap:
        return None
    csi = snap.get("csi")
    if csi is None:
        return None
    cnn_csi, cnn_serial_pair = key_state.current_cnn_csi()
    return (
        snap.get("serial"),
        np.asarray(csi, dtype=np.float32).copy(),
        key_state.current_epoch(),
        cnn_csi,
        cnn_serial_pair,
        snap.get("rssi")
    )


# ======================================================
# CSI Watcher (UAV)
# ======================================================
class CSISerialWatcher:
    def __init__(self, port, baudrate):
        self._lock = threading.Lock()
        self._latest = {}
        self._samples_by_serial = {}

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

    def samples(self):
        with self._lock:
            return {
                serial: {
                    **sample,
                    "csi": sample["csi"].copy(),
                }
                for serial, sample in self._samples_by_serial.items()
            }

    def force_reset(self):
        with self._lock:
            self._latest.clear()
            self._samples_by_serial.clear()
        self.streamer.force_reset()

    def _handle(self, raw):
        sample = self._parse(raw)
        if sample:
            with self._lock:
                self._latest["UAV"] = sample
                self._samples_by_serial[sample["serial"]] = sample
                if len(self._samples_by_serial) > 512:
                    overflow = len(self._samples_by_serial) - 512
                    for serial in sorted(self._samples_by_serial)[:overflow]:
                        self._samples_by_serial.pop(serial, None)

    def _parse(self, line):
        sample = preprocess_csi_line(line)
        if not sample or sample.get("device") != "UAV":
            return None
        return {
            "serial": sample["serial"],
            "csi": sample["csi"],
            "rssi": sample["rssi"],
        }


# ======================================================
# CNN / CNN-Q reconstruction
# ======================================================
def reconstruct_csi_cnn(model_csi, csi_1, csi_2) -> np.ndarray:
    """
    PLKG.py-style CSI model input:
    input_array shape (1, 2, 51) -> unsqueeze(0) -> (1, 1, 2, 51).
    """
    csi_1 = np.asarray(csi_1, dtype=np.float32).reshape(-1)
    csi_2 = np.asarray(csi_2, dtype=np.float32).reshape(-1)
    if csi_1.size != 51 or csi_2.size != 51:
        raise ValueError(
            f"CSI CNN input size mismatch: expected 51+51, got {csi_1.size}+{csi_2.size}"
        )

    c1 = csi_1.reshape(1, 1, 1, 51)
    c2 = csi_2.reshape(1, 1, 1, 51)
    model_input = np.concatenate([c1, c2], axis=2)  # (1, 1, 2, 51)

    with torch.no_grad():
        out = model_csi(torch.from_numpy(model_input).float())

    features = out.detach().cpu().numpy().reshape(-1)
    if features.size != 51:
        raise ValueError(f"CSI CNN output size mismatch: expected 51, got {features.size}")
    return features.astype(np.float32)


def _bits_to_float_array(bits: str, expected_len: int) -> np.ndarray:
    bits = force_102_bits(bits) if expected_len == 102 else str(bits)
    arr = np.array([int(bit) for bit in bits], dtype=np.float32)
    if arr.size != expected_len:
        raise ValueError(f"bit array size mismatch: expected {expected_len}, got {arr.size}")
    return arr


def reconstruct_key_cnnq(model_q, raw_bits: str, peer_bits: str | None = None, debug: bool = False) -> str:
    """
    PLKG.py-style CNN-Q model input:
    [key1, key2] shape (2, 102) -> unsqueeze(0) -> (1, 2, 102).

    raw_bits: 102-bit string
    peer_bits: second consecutive 102-bit string, matching PLKG.py's key pair input
    return: reconstructed 102-bit string
    """
    arr = _bits_to_float_array(raw_bits, 102)
    peer_arr = arr if peer_bits is None else _bits_to_float_array(peer_bits, 102)

    key_pair = np.stack([arr, peer_arr], axis=0)   # (2, 102)
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

    bits = np.rint(out_flat).clip(0, 1).astype(np.int32)

    return "".join(str(int(b)) for b in bits)


def hamming_distance(a: str, b: str) -> int | None:
    if not a or not b or len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


def quantize_csi_direct(csi) -> str:
    features = np.asarray(csi, dtype=np.float32).reshape(-1)
    if features.size != 51:
        raise ValueError(f"UAV CSI feature size mismatch: expected 51, got {features.size}")
    return force_102_bits(quantization_1(features, Nbits=2, inbits=13, guard=0))


def select_latest_consecutive_pair(samples, last_pair_start=None):
    serials = set(samples)
    candidates = [
        serial
        for serial in serials
        if serial + 1 in serials and (last_pair_start is None or serial > last_pair_start)
    ]
    if not candidates:
        return None
    start = max(candidates)
    return start, start + 1

# ======================================================
# THREAD 1: Key generation + CNN-Q + BCH
# ======================================================
def keygen_thread():
    global uav_csi_watcher
    model_csi = None
    model_q = None
    try:
        model_csi = torch.load(MODEL_CSI_PATH, map_location="cpu", weights_only=False)
        model_csi.eval()
        print(f"[UAV] CSI CNN model loaded: {MODEL_CSI_PATH}")
    except Exception as exc:
        print(f"[UAV] CSI CNN diagnostic disabled: {exc}")

    try:
        model_q = torch.load(MODEL_KEY_QUAN_PATH, map_location="cpu", weights_only=False)
        model_q.eval()
        print(f"[UAV] CNN-Q model loaded: {MODEL_KEY_QUAN_PATH}")
        print("[UAV] CNN-Q live input uses two consecutive UAV raw keys, aligned with PLKG.py.")
    except Exception as exc:
        print(f"[UAV] CNN-Q diagnostic disabled: {exc}")

    watcher = CSISerialWatcher(CSI_PORT, CSI_BAUD)
    uav_csi_watcher = watcher
    watcher.start()

    last_serial = None
    last_pair_start = None
    next_keygen_time = 0.0
    last_wait_log = 0.0
    consecutive_pair_wait_logs = 0
    last_forced_csi_reset = 0.0
    if KEY_SOURCE not in ("csi", "cnn", "cnnq"):
        print(f"[UAV] unknown PLKG_KEY_SOURCE={KEY_SOURCE!r}; falling back to cnnq")
        key_source = "cnnq"
    else:
        key_source = KEY_SOURCE

    print(
        f"[UAV] keygen started (BCH target={key_source}, "
        f"CNN={'on' if model_csi is not None else 'off'}, "
        f"CNN-Q={'on' if model_q is not None else 'off'})"
    )

    while True:
        if keygen_resync_event.is_set():
            keygen_resync_event.clear()
            last_serial = None
            last_pair_start = None
            next_keygen_time = 0.0
            consecutive_pair_wait_logs = 0
            last_wait_log = 0.0
            print("[UAV] keygen resync requested; waiting for fresh CSI")

        expired_pending = key_state.expire_pending(KEY_ACK_TIMEOUT_SEC)
        if expired_pending is not None:
            print(
                f"[UAV] pending key epoch={expired_pending['epoch']} "
                f"serial_pair={expired_pending['serial_token']} timed out before GSN ACK; "
                "keeping previous active video key"
            )

        if key_state.has_pending():
            time.sleep(0.05)
            continue

        samples = watcher.samples()
        pair = select_latest_consecutive_pair(samples, last_pair_start)
        if pair is None:
            now = time.monotonic()
            if now - last_wait_log >= CSI_PAIR_WAIT_LOG_INTERVAL_SEC:
                consecutive_pair_wait_logs += 1
                print(
                    "[UAV] waiting for two consecutive local CSI samples before key generation "
                    f"(wait_count={consecutive_pair_wait_logs}/{CSI_PAIR_WAIT_RESET_LOGS}, "
                    f"buffered_samples={len(samples)})"
                )
                last_wait_log = now

                reset_due = consecutive_pair_wait_logs >= CSI_PAIR_WAIT_RESET_LOGS
                reset_cooled_down = now - last_forced_csi_reset >= CSI_PAIR_WAIT_RESET_COOLDOWN_SEC
                if reset_due and reset_cooled_down:
                    print(
                        "[UAV] no consecutive CSI samples after repeated waits; "
                        f"forcing ESP32 CSI reset on {CSI_PORT}"
                    )
                    try:
                        watcher.force_reset()
                        last_serial = None
                        last_pair_start = None
                        next_keygen_time = 0.0
                        consecutive_pair_wait_logs = 0
                        last_forced_csi_reset = now
                    except Exception as exc:
                        print(f"[UAV] failed to force reset ESP32 CSI reader: {exc}")
            time.sleep(0.05)
            continue
        consecutive_pair_wait_logs = 0

        serial_1, serial_2 = pair
        s1 = samples[serial_1]
        s2 = samples[serial_2]
        last_serial = serial_2
        last_pair_start = serial_1

        # === Step 1: CSI direct quantization, aligned with GSN local keygen ===
        csi_1 = s1["csi"]
        csi_2 = s2["csi"]
        direct_key = quantize_csi_direct(csi_1)
        direct_key_2 = quantize_csi_direct(csi_2)

        cnn_csi = None
        cnn_key = None
        cnnq_key = None
        if model_csi is not None:
            try:
                # === Step 2: CSI -> CNN -> corrected CSI -> CNN key diagnostic ===
                cnn_csi = reconstruct_csi_cnn(model_csi, csi_1, csi_2)
                key_state.update_live_cnn_csi((serial_1, serial_2), cnn_csi)
                bits = quantization_1(cnn_csi, Nbits=2, inbits=13, guard=0)
                cnn_key = force_102_bits("".join(str(b) for b in bits))
            except Exception as exc:
                print(f"[UAV] CNN diagnostic failed for serial_pair={serial_1},{serial_2}: {exc}")
                cnn_csi = None
                cnn_key = None

        now = time.monotonic()
        if now < next_keygen_time:
            time.sleep(0.01)
            continue

        if model_q is not None:
            try:
                # === Step 3: direct key -> CNN-Q -> corrected key diagnostic ===
                cnnq_key = reconstruct_key_cnnq(model_q, direct_key, peer_bits=direct_key_2, debug=DEBUG)
            except Exception as exc:
                print(f"[UAV] CNN-Q diagnostic failed for serial_pair={serial_1},{serial_2}: {exc}")
                cnnq_key = None

        if key_source == "cnnq":
            if cnnq_key is None:
                print("[UAV] cnnq key source requested but CNN-Q output is unavailable; skipping epoch")
                continue
            target_key = cnnq_key
        elif key_source == "cnn":
            if cnn_key is None:
                print("[UAV] cnn key source requested but CNN output is unavailable; skipping epoch")
                continue
            target_key = cnn_key
        else:
            target_key = direct_key

        # === Step 4: BCH syndrome helper for reconciliation ===
        helper = bch_encode_syndrome_b64(target_key)

        # === Step 5: privacy amplification / AES-256 key ===
        aes_key = sha256.sha_byte(target_key)

        # === Step 6: stage pending key; video keeps using the previous active key until GSN ACK ===
        pending_epoch = key_state.stage_pending(
            (serial_1, serial_2),
            helper,
            aes_key,
            rssi=s1.get("rssi"),
            raw_csi=csi_1,
            raw_csi_2=csi_2,
            raw_key=direct_key,
            raw_key_2=direct_key_2,
            cnn_csi=cnn_csi,
            cnn_key=cnn_key,
            cnnq_key=cnnq_key,
            corrected_key=target_key,
        )
        if pending_epoch is None:
            time.sleep(0.05)
            continue
        next_keygen_time = time.monotonic() + KEY_UPDATE_INTERVAL_SEC
        direct_to_cnn = hamming_distance(direct_key, cnn_key)
        direct_to_cnnq = hamming_distance(direct_key, cnnq_key)
        diag = ""
        if direct_to_cnn is not None and direct_to_cnnq is not None:
            diag = f" direct-vs-cnn={direct_to_cnn}/102 direct-vs-cnnq={direct_to_cnnq}/102"
        print(
            f"[UAV] pending key epoch={pending_epoch} "
            f"serial_pair={serial_1},{serial_2} source={key_source}{diag} "
            "waiting for GSN ACK before switching video key"
        )


def control_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UAV_CONTROL_PORT))
    print(f"[UAV] control channel listening on UDP/{UAV_CONTROL_PORT}")

    while True:
        data, addr = sock.recvfrom(1024)
        cmd = data.decode(errors="ignore").strip()

        if cmd.startswith("KEY_ACK"):
            parts = cmd.split()
            if len(parts) != 4:
                print(f"[UAV] invalid KEY_ACK command from {addr[0]}: {cmd!r}")
                continue
            try:
                epoch = int(parts[1])
            except ValueError:
                print(f"[UAV] invalid KEY_ACK epoch from {addr[0]}: {parts[1]!r}")
                continue
            ok, message = key_state.activate_pending_ack(epoch, parts[2], parts[3])
            if ok:
                print(f"[UAV] GSN confirmed pending key; {message}")
            else:
                print(f"[UAV] ignored KEY_ACK from {addr[0]}: {message}")
            continue

        if cmd.startswith("VIDEO_ENCRYPTION"):
            parts = cmd.split()
            if len(parts) < 2:
                print(f"[UAV] invalid VIDEO_ENCRYPTION command from {addr[0]}: {cmd!r}")
                continue
            value = parts[1].strip().lower()
            if value in ("1", "on", "true", "enabled", "encrypt", "encrypted"):
                enabled = True
            elif value in ("0", "off", "false", "disabled", "plain", "plaintext"):
                enabled = False
            else:
                print(f"[UAV] invalid VIDEO_ENCRYPTION value from {addr[0]}: {value!r}")
                continue
            changed = video_encryption_state.set_enabled(enabled)
            state = "ENCRYPTED" if enabled else "PLAINTEXT"
            suffix = "" if changed else " (unchanged)"
            print(f"[UAV] video transmission mode set to {state} by {addr[0]}{suffix}")
            continue

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
    args = parse_args()
    gsn_ip = args.gsn_ip
    print(f"[UAV] GSN target IP: {gsn_ip}")
    print(
        f"[UAV] video config: resolution={args.video_resolution}, fps={args.video_fps}, "
        f"bitrate={args.video_bitrate}, iperiod={args.video_iperiod}, "
        f"chunk={args.video_chunk}, "
        f"exposure={'auto' if args.fixed_exposure_us is None else args.fixed_exposure_us}, "
        f"gain={'auto' if args.analogue_gain is None else args.analogue_gain}, "
        f"codec={'software JPEG' if args.software_jpeg else 'hardware H.264'}"
    )

    # Key generation thread
    threading.Thread(target=keygen_thread, daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()

    # Reconciliation helper sender
    sender = UAVKeySender(key_state.for_reconciliation, gsn_ip, debug=DEBUG)
    threading.Thread(target=sender.run, daemon=True).start()

    if DEMO_TELEMETRY_ENABLED:
        demo_sender = DemoTelemetrySender(
            key_state.for_demo_telemetry,
            gsn_ip,
            port=DEMO_TELEMETRY_PORT,
            debug=DEBUG,
        )
        threading.Thread(target=demo_sender.run, daemon=True).start()
        if LiveCSITelemetrySender is not None:
            live_csi_sender = LiveCSITelemetrySender(
                latest_uav_csi_for_telemetry,
                gsn_ip,
                port=DEMO_TELEMETRY_PORT,
                debug=DEBUG,
                send_interval=LIVE_CSI_TELEMETRY_INTERVAL_SEC,
            )
            threading.Thread(target=live_csi_sender.run, daemon=True).start()
        print(
            f"[UAV] demo telemetry enabled on UDP/{DEMO_TELEMETRY_PORT}; "
            f"live CSI interval={LIVE_CSI_TELEMETRY_INTERVAL_SEC:.3f}s"
        )
    elif DEMO_TELEMETRY_IMPORT_ERROR is not None:
        print(f"[UAV] demo telemetry disabled: {DEMO_TELEMETRY_IMPORT_ERROR}")

    # Video streamer (preview in main thread is more stable)
    streamer = UAVVideoStreamer(
        key_state.for_video,
        gsn_ip,
        preview=args.preview or PREVIEW,
        debug=DEBUG,
        resolution=args.video_resolution,
        fps=args.video_fps,
        h264_bitrate=args.video_bitrate,
        h264_iperiod=args.video_iperiod,
        jpeg_quality=args.jpeg_quality,
        chunk=args.video_chunk,
        flip_code=VIDEO_FLIP_CODE,
        fixed_exposure_us=args.fixed_exposure_us,
        analogue_gain=args.analogue_gain,
        use_hardware_h264=VIDEO_USE_HARDWARE_H264 and not args.software_jpeg,
        sync_port=TIME_SYNC_PORT,
        sync_samples=TIME_SYNC_SAMPLES,
        get_encryption_enabled=video_encryption_state.is_enabled,
    )

    print("[UAV] system ready (press q to quit preview)")
    streamer.run()
