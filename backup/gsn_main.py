# gsn_main.py  (FINAL, memory-only, epoch-ready)

import threading
import time
import socket
import cv2

import sha256
from csi_control import send_csi_reset_request
from key_confirm import verify_key_confirm
from rs_ecc import rs_decode_key
from gsn_key_generate import load_model, CSISerialWatcher, generate_key
from gsn_receiver import GSNReceiver
from gsn_key_matcher import LiveKDRPlotter

UAV_CONTROL_PORT = 5008


# ---------------- Key State ----------------
key_lock = threading.Lock()
gsn_raw = None
gsn_raw_by_serial = {}
keys_by_epoch = {}   # epoch -> aes_key
RAW_HISTORY_LIMIT = 512


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
    global gsn_raw, gsn_raw_by_serial
    model = load_model("model_reserved/cnn_basic/model_final.pth")
    # watcher = CSISerialWatcher("/dev/ttyUSB0", 115200)
    watcher = CSISerialWatcher("/dev/tty.usbserial-0001", 115200)  # macOS
    # watcher = CSISerialWatcher("COM3", 115200)  # Windows
    watcher.start()

    last_serial = None
    while True:
        s = watcher.snapshot().get("GSN")
        if not s:
            continue
        if s["serial"] == last_serial:
            continue
        last_serial = s["serial"]

        raw, _ = generate_key(model, s["csi"])
        with key_lock:
            gsn_raw = raw
            gsn_raw_by_serial[s["serial"]] = raw
            if len(gsn_raw_by_serial) > RAW_HISTORY_LIMIT:
                oldest = sorted(gsn_raw_by_serial)[: len(gsn_raw_by_serial) - RAW_HISTORY_LIMIT]
                for serial in oldest:
                    gsn_raw_by_serial.pop(serial, None)


# ---------------- RS Receiver Thread ----------------
def rs_thread(plotter):
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
        serial  = int(parts[2])
        parity  = parts[3]
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
        if local_raw is None:
            print(f"[GSN] waiting for local CSI serial={serial} for epoch={epoch}")
            continue

        try:
            corrected = rs_decode_key(local_raw, parity)
        except ValueError as exc:
            print(f"[GSN] RS correction failed for epoch={epoch}: {exc}")
            continue
        aes = sha256.sha_byte(corrected)
        if not verify_key_confirm(aes, epoch, serial, parity, confirm):
            rejected_epochs.add(epoch)
            print(f"[GSN] key confirmation failed for epoch={epoch}, serial={serial}")
            continue

        with key_lock:
            keys_by_epoch[epoch] = aes
            print(
                f"[KEY ACTIVE] epoch={epoch} | confirmed | AES={aes.hex()[:32]}..."
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

    # Start background key generation and RS reconciliation threads.
    threading.Thread(
        target=keygen_thread,
        daemon=True
    ).start()

    threading.Thread(
        target=rs_thread,
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
