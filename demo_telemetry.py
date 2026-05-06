import json
import socket
import time


DEMO_TELEMETRY_PORT = 5009
DEMO_TELEMETRY_TYPE = "DEMO_KEY_SNAPSHOT"
LIVE_CSI_TELEMETRY_TYPE = "LIVE_CSI_SNAPSHOT"


def _as_float_list(values, precision=6):
    if values is None:
        return None
    out = []
    for value in values:
        try:
            out.append(round(float(value), precision))
        except (TypeError, ValueError):
            continue
    return out


def make_demo_packet(
    epoch,
    serial,
    raw_csi,
    raw_key,
    corrected_key,
    cnn_csi=None,
    cnn_key=None,
    cnnq_key=None,
    serial_pair=None,
    raw_csi_2=None,
    raw_key_2=None,
):
    if serial_pair is None:
        serial_pair = [int(serial)]
    else:
        serial_pair = [int(item) for item in serial_pair]

    packet = {
        "type": DEMO_TELEMETRY_TYPE,
        "epoch": int(epoch),
        "serial": int(serial),
        "serial_pair": serial_pair,
        "uav_raw_csi": _as_float_list(raw_csi),
        "uav_raw_csi_2": _as_float_list(raw_csi_2),
        "uav_raw_key": str(raw_key),
        "uav_raw_key_2": None if raw_key_2 is None else str(raw_key_2),
        "uav_cnn_csi": _as_float_list(cnn_csi),
        "uav_cnn_key": None if cnn_key is None else str(cnn_key),
        "uav_cnnq_key": None if cnnq_key is None else str(cnnq_key),
        "uav_corrected_key": str(corrected_key),
    }
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def make_live_csi_packet(serial, csi, epoch=None):
    packet = {
        "type": LIVE_CSI_TELEMETRY_TYPE,
        "serial": int(serial),
        "uav_live_csi": _as_float_list(csi),
        "epoch": None if epoch is None else int(epoch),
    }
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def parse_demo_packet(data):
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if payload.get("type") != DEMO_TELEMETRY_TYPE:
        return None

    try:
        epoch = int(payload["epoch"])
        serial = int(payload["serial"])
        serial_pair = payload.get("serial_pair")
        if serial_pair is None:
            serial_pair = [serial]
        serial_pair = tuple(int(item) for item in serial_pair)
        uav_raw_csi = _as_float_list(payload.get("uav_raw_csi"))
        uav_raw_csi_2 = _as_float_list(payload.get("uav_raw_csi_2"))
        uav_raw_key = str(payload["uav_raw_key"])
        uav_raw_key_2 = payload.get("uav_raw_key_2")
        uav_corrected_key = str(payload["uav_corrected_key"])
        uav_cnn_csi = _as_float_list(payload.get("uav_cnn_csi"))
        uav_cnn_key = payload.get("uav_cnn_key")
        uav_cnnq_key = payload.get("uav_cnnq_key")
    except (KeyError, TypeError, ValueError):
        return None

    if not uav_raw_csi or not uav_raw_key or not uav_corrected_key:
        return None

    return {
        "epoch": epoch,
        "serial": serial,
        "serial_pair": serial_pair,
        "uav_raw_csi": uav_raw_csi,
        "uav_raw_csi_2": uav_raw_csi_2,
        "uav_raw_key": uav_raw_key,
        "uav_raw_key_2": None if uav_raw_key_2 is None else str(uav_raw_key_2),
        "uav_cnn_csi": uav_cnn_csi,
        "uav_cnn_key": None if uav_cnn_key is None else str(uav_cnn_key),
        "uav_cnnq_key": None if uav_cnnq_key is None else str(uav_cnnq_key),
        "uav_corrected_key": uav_corrected_key,
        "time": time.time(),
    }


def parse_live_csi_packet(data):
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if payload.get("type") != LIVE_CSI_TELEMETRY_TYPE:
        return None

    try:
        serial = int(payload["serial"])
        uav_live_csi = _as_float_list(payload.get("uav_live_csi"))
        epoch = payload.get("epoch")
        if epoch is not None:
            epoch = int(epoch)
    except (KeyError, TypeError, ValueError):
        return None

    if not uav_live_csi:
        return None

    return {
        "packet_type": "live_csi",
        "serial": serial,
        "epoch": epoch,
        "uav_live_csi": uav_live_csi,
        "time": time.time(),
    }


def parse_telemetry_packet(data):
    live = parse_live_csi_packet(data)
    if live:
        return live

    demo = parse_demo_packet(data)
    if demo:
        demo["packet_type"] = "key_snapshot"
    return demo


class DemoTelemetrySender:
    def __init__(
        self,
        get_demo_state,
        gsn_ip,
        port=DEMO_TELEMETRY_PORT,
        debug=False,
        resend_interval=1.0,
        new_epoch_burst=3,
    ):
        """
        get_demo_state() -> (
            epoch,
            csi_serial,
            raw_csi,
            raw_key,
            corrected_key,
            cnn_csi,
            cnn_key,
            cnnq_key,
            serial_pair,
            raw_csi_2,
            raw_key_2,
        )
        """
        self.get_demo_state = get_demo_state
        self.gsn_ip = gsn_ip
        self.port = port
        self.debug = debug
        self.resend_interval = resend_interval
        self.new_epoch_burst = new_epoch_burst

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        last_epoch = -1
        last_send_time = 0.0

        while True:
            state = self.get_demo_state()
            epoch, serial, raw_csi, raw_key, corrected_key = state[:5]
            cnn_csi = state[5] if len(state) > 5 else None
            cnn_key = state[6] if len(state) > 6 else None
            cnnq_key = state[7] if len(state) > 7 else None
            serial_pair = state[8] if len(state) > 8 else None
            raw_csi_2 = state[9] if len(state) > 9 else None
            raw_key_2 = state[10] if len(state) > 10 else None
            now = time.time()
            if (
                epoch < 0
                or serial is None
                or raw_csi is None
                or raw_key is None
                or corrected_key is None
            ):
                time.sleep(0.05)
                continue

            should_send = False
            repeat_count = 1

            if epoch != last_epoch:
                should_send = True
                repeat_count = self.new_epoch_burst
            elif now - last_send_time >= self.resend_interval:
                should_send = True

            if not should_send:
                time.sleep(0.05)
                continue

            msg = make_demo_packet(
                epoch,
                serial,
                raw_csi,
                raw_key,
                corrected_key,
                cnn_csi=cnn_csi,
                cnn_key=cnn_key,
                cnnq_key=cnnq_key,
                serial_pair=serial_pair,
                raw_csi_2=raw_csi_2,
                raw_key_2=raw_key_2,
            )
            for _ in range(repeat_count):
                sock.sendto(msg, (self.gsn_ip, self.port))
                time.sleep(0.01)
            if self.debug:
                print(
                    f"[UAV] send demo telemetry epoch={epoch} "
                    f"serial={serial} to {self.gsn_ip}:{self.port}"
                )

            last_epoch = epoch
            last_send_time = time.time()


class LiveCSITelemetrySender:
    def __init__(
        self,
        get_csi_state,
        gsn_ip,
        port=DEMO_TELEMETRY_PORT,
        debug=False,
        send_interval=0.05,
    ):
        """
        get_csi_state() -> (serial, csi, epoch)
        Sends the latest UAV CSI for visualization only. This is intentionally
        independent from the key epoch/helper sender.
        """
        self.get_csi_state = get_csi_state
        self.gsn_ip = gsn_ip
        self.port = port
        self.debug = debug
        self.send_interval = send_interval

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        last_serial = None
        last_send_time = 0.0
        last_log_time = 0.0

        while True:
            state = self.get_csi_state()
            if not state:
                time.sleep(0.02)
                continue

            serial, csi, epoch = state
            now = time.time()
            if serial is None or csi is None:
                time.sleep(0.02)
                continue

            if serial == last_serial or now - last_send_time < self.send_interval:
                time.sleep(0.005)
                continue

            msg = make_live_csi_packet(serial, csi, epoch=epoch)
            sock.sendto(msg, (self.gsn_ip, self.port))

            if self.debug and now - last_log_time >= 1.0:
                print(
                    f"[UAV] send live CSI serial={serial} "
                    f"to {self.gsn_ip}:{self.port}"
                )
                last_log_time = now

            last_serial = serial
            last_send_time = now
