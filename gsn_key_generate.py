# gsn_key_generate.py  (LIBRARY, memory-only)

import threading
import time
import numpy as np
import torch

from greycode_quantization import quantization_1
import sha256
from data_collecting_processing.collect import CSISerialStreamer
from models.cnn_basic import cnn_basic
torch.serialization.add_safe_globals([cnn_basic])

NO_USED_CARRIERS_CH1 = [0, 1, 2, 3, 4, 5, 11, 32, 59, 60, 61, 62, 63]
NO_USED_CARRIERS_CH13 = [0, 1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]


def load_model(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
    else:
        model = cnn_basic()
        model.load_state_dict(obj.get("state_dict", obj), strict=False)
    model.eval()
    return model


def load_cnnq_model(path):
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def reconstruct_key_cnnq(model_q, raw_bits: str) -> str:
    arr = np.array([int(b) for b in raw_bits], dtype=np.float32)
    key_pair = np.stack([arr, arr], axis=0).reshape(1, 2, 102)

    with torch.no_grad():
        out = model_q(torch.from_numpy(key_pair))

    out_flat = out.detach().cpu().numpy().reshape(-1)
    if out_flat.size != 102:
        raise ValueError(
            f"CNN-Q output size mismatch: expected 102, got {out_flat.size}"
        )

    bits = (out_flat > 0.5).astype(np.int32)
    return "".join(str(int(b)) for b in bits)


class CSISerialWatcher:
    def __init__(self, port, baudrate):
        self._lock = threading.Lock()
        self._latest = {}

        self.streamer = CSISerialStreamer(
            port, baudrate,
            endpoint_type="GSN",
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
                self._latest["GSN"] = sample

    def _parse(self, line):
        if "serial_num:" not in line:
            return None
        try:
            parts = [p for p in line.split(",") if p != ""]
            if parts[0] != "serial_num:":
                return None

            serial = int(parts[1])
            rssi = float(parts[3])
            noise = float(parts[4])
            tail = [float(x) for x in parts[5:] if x]

            amps = np.array(tail[:64], dtype=np.float32)
            amps = np.delete(amps, NO_USED_CARRIERS_CH13)
            if len(amps) != 51:
                return None

            return {
                "serial": serial,
                "csi": amps / np.max(amps),
                "rssi": rssi,
                "noise": noise,
                "time": time.time(),
            }
        except:
            return None


def generate_key(model, csi):
    c1 = csi.reshape(1,1,1,51)
    c2 = np.zeros_like(c1)
    arr = np.concatenate([c1,c2], axis=2)
    arr = np.concatenate([arr, arr], axis=0)

    with torch.no_grad():
        out = model(torch.from_numpy(arr).float())

    features = out[0].cpu().numpy().ravel()
    key_bits = quantization_1(features, Nbits=2, inbits=13, guard=0)
    aes_key = sha256.sha_byte(key_bits)

    return "".join(str(b) for b in key_bits), aes_key
