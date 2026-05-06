# gsn_key_generate.py  (LIBRARY, memory-only)
#
# GSN side reads local CSI and generates a noisy local raw key.
# UAV owns the CNN/CNN-Q correction flow and sends BCH helper data.

import threading
import time
import numpy as np

from greycode_quantization import quantization_1
import sha256
from data_collecting_processing.collect import CSISerialStreamer
from fetch_ESP32_CSI import preprocess_csi_line


class CSISerialWatcher:
    def __init__(self, port, baudrate, endpoint_type="GSN", device="GCS"):
        self._lock = threading.Lock()
        self._latest = {}
        self.endpoint_type = endpoint_type
        self.device = device
        self._local_serial = 0

        self.streamer = CSISerialStreamer(
            port, baudrate,
            endpoint_type=endpoint_type,
            write_file=False,
            callback=self._handle
        )

    def start(self):
        self.streamer.start()

    def snapshot(self):
        with self._lock:
            return {k: v.copy() for k, v in self._latest.items()}

    def _handle(self, raw):
        sample = self._parse(raw)
        if sample:
            with self._lock:
                self._latest[self.endpoint_type] = sample

    def _parse(self, line):
        sample = preprocess_csi_line(line)
        if not sample or sample.get("device") != self.device:
            return None
        serial = sample["serial"]
        if self.device == "EVE" and serial == 0:
            self._local_serial += 1
            serial = self._local_serial
        parsed = {
            "serial": serial,
            "device": sample["device"],
            "csi": sample["csi"],
            "raw_csi": sample["raw_csi"],
            "rssi": sample["rssi"],
            "noise": sample["noise"],
            "time": time.time(),
        }
        if "mac" in sample:
            parsed["mac"] = sample["mac"]
        return parsed


def generate_key(csi, return_features=False):
    features = np.asarray(csi, dtype=np.float32).reshape(-1)
    if features.size != 51:
        raise ValueError(f"GSN CSI feature size mismatch: expected 51, got {features.size}")

    key_bits = quantization_1(features, Nbits=2, inbits=13, guard=0)
    aes_key = sha256.sha_byte(key_bits)

    key = "".join(str(b) for b in key_bits)
    if return_features:
        return key, aes_key, features.copy()
    return key, aes_key
