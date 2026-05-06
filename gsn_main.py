# gsn_main.py  (FINAL, memory-only, epoch-ready)

import threading
import time
import socket
import cv2

import sha256
from csi_control import send_csi_reset_request
from key_confirm import verify_key_confirm
from bch_reconciliation import bch_decode_key
from gsn_key_generate import CSISerialWatcher, generate_key
from gsn_receiver import GSNReceiver
from gsn_key_matcher import LiveKDRPlotter

UAV_CONTROL_PORT = 5008
GSN_CSI_PORT = "/dev/tty.usbserial-0001"
EVE_CSI_PORT = "/dev/cu.usbserial-4"
CSI_BAUD = 115200


# ---------------- Key State ----------------
key_lock = threading.Lock()
gsn_raw = None
gsn_raw_by_serial = {}
eve_latest = None
keys_by_epoch = {}   # epoch -> aes_key
RAW_HISTORY_LIMIT = 512


def parse_serial_pair(value):
    text = str(value).strip()
    if "," in text:
        return tuple(int(item) for item in text.split(",") if item != "")
    if "-" in text:
        return tuple(int(item) for item in text.split("-") if item != "")
    return (int(text),)


def serial_pair_label(pair):
    return ",".join(str(item) for item in pair)


def show_frame(frame, latency):
    if frame is None:
        return
    cv2.putText(
        frame,
        f"Latency={latency:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.imshow("GSN Video", frame)
    cv2.waitKey(1)


def get_key(epoch):
    with key_lock:
        return keys_by_epoch.get(epoch)


# ---------------- GSN KeyGen Thread ----------------
def keygen_thread():
    global gsn_raw, gsn_raw_by_serial, eve_latest
    # watcher = CSISerialWatcher("/dev/ttyUSB0", 115200)
    watcher = CSISerialWatcher(GSN_CSI_PORT, CSI_BAUD)  # macOS
    # watcher = CSISerialWatcher("COM3", 115200)  # Windows
    watcher.start()

    eve_watcher = None
    try:
        eve_watcher = CSISerialWatcher(EVE_CSI_PORT, CSI_BAUD, endpoint_type="EVE", device="EVE")
        eve_watcher.start()
        print(f"[GSN] EVE CSI watcher started on {EVE_CSI_PORT}")
    except Exception as exc:
        print(f"[GSN] failed to start EVE CSI watcher on {EVE_CSI_PORT}: {exc}")

    last_serial = None
    last_eve_serial = None
    while True:
        if eve_watcher is not None:
            eve = eve_watcher.snapshot().get("EVE")
            if eve and eve["serial"] != last_eve_serial:
                last_eve_serial = eve["serial"]
                with key_lock:
                    eve_latest = eve
                if eve["serial"] % 20 == 0:
                    print(
                        f"[GSN] read EVE CSI seq={eve['serial']} "
                        f"rssi={eve['rssi']:.1f} noise={eve['noise']:.1f}"
                    )

        s = watcher.snapshot().get("GSN")
        if not s:
            time.sleep(0.01)
            continue
        if s["serial"] == last_serial:
            time.sleep(0.002)
            continue
        last_serial = s["serial"]

        raw, _ = generate_key(s["csi"])
        with key_lock:
            gsn_raw = raw
            gsn_raw_by_serial[s["serial"]] = raw
            if len(gsn_raw_by_serial) > RAW_HISTORY_LIMIT:
                oldest = sorted(gsn_raw_by_serial)[: len(gsn_raw_by_serial) - RAW_HISTORY_LIMIT]
                for serial in oldest:
                    gsn_raw_by_serial.pop(serial, None)


# ---------------- BCH Receiver Thread ----------------
def bch_thread(plotter):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5007))
    last_epoch = -1
    pending_uav_csi_reset = True
    rejected_epochs = set()

    while True:
        data, addr = sock.recvfrom(65535)
        parts = data.decode().split()
        if len(parts) != 5 or parts[0] != "R":
            continue

        epoch   = int(parts[1])
        serial_token = parts[2]
        serial_pair = parse_serial_pair(serial_token)
        serial = serial_pair[0]
        peer_serial = serial_pair[1] if len(serial_pair) > 1 else serial
        helper  = parts[3]
        confirm = parts[4]

        if pending_uav_csi_reset:
            reset_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                for _ in range(3):
                    reset_sock.sendto(b"RESET_CSI", (addr[0], UAV_CONTROL_PORT))
                    time.sleep(0.05)
            finally:
                reset_sock.close()
            with key_lock:
                keys_by_epoch.clear()
            print(f"[GSN] Sent RESET_CSI to UAV at {addr[0]}")
            pending_uav_csi_reset = False
            last_epoch = -1
            rejected_epochs.clear()
            continue

        if last_epoch >= 0 and epoch < last_epoch:
            with key_lock:
                keys_by_epoch.clear()
            print(f"[GSN] UAV key session reset detected: epoch {last_epoch}->{epoch}")
            rejected_epochs.clear()
        last_epoch = epoch

        with key_lock:
            if epoch in keys_by_epoch:
                continue
        if epoch in rejected_epochs:
            continue
       
        with key_lock:
            local_raw = gsn_raw_by_serial.get(serial)
            local_peer_raw = gsn_raw_by_serial.get(peer_serial)
        if local_raw is None or local_peer_raw is None:
            print(f"[GSN] waiting for local CSI serials={serial_pair_label(serial_pair)} for epoch={epoch}")
            continue

        try:
            corrected = bch_decode_key(local_raw, helper)
        except ValueError as exc:
            print(f"[GSN] BCH correction failed for epoch={epoch}: {exc}")
            continue
        aes = sha256.sha_byte(corrected)
        if not verify_key_confirm(aes, epoch, serial_token, helper, confirm):
            rejected_epochs.add(epoch)
            print(f"[GSN] key confirmation failed for epoch={epoch}, serials={serial_pair_label(serial_pair)}")
            continue

        with key_lock:
            keys_by_epoch[epoch] = aes
            print(
                f"[KEY ACTIVE] epoch={epoch} serials={serial_pair_label(serial_pair)} "
                f"| confirmed | AES={aes.hex()[:32]}..."
            )

        

        plotter.update(local_raw, corrected)


# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        send_csi_reset_request(port=UAV_CONTROL_PORT)
        print("[GSN] sent RESET_CSI request to UAV")
    except Exception as exc:
        print(f"[GSN] failed to send RESET_CSI request: {exc}")
    # 1️⃣ 建立 plotter（但不要啟動）
    plotter = LiveKDRPlotter()

    # Start background key generation and BCH reconciliation threads.
    threading.Thread(
        target=keygen_thread,
        daemon=True
    ).start()

    threading.Thread(
        target=bch_thread,
        args=(plotter,),
        daemon=True
    ).start()

    # 3️⃣ 啟動 video receiver（它自己內部開 thread）
    rx = GSNReceiver(
        get_aes_key=get_key,
        on_frame=show_frame
    )
    rx.start()   # ✔ 這不是 thread，是 class 啟動，OK

    # 4️⃣ 最後一行：main thread 專門跑 GUI
    plotter.start()   # ✅ 一定要在 main thread
