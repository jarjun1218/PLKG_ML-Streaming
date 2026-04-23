# gsn_main.py  (FINAL, memory-only, epoch-ready)

import threading
import time
import socket
import sha256
import cv2

from rs_ecc import rs_decode_key
from gsn_key_generate import load_model, CSISerialWatcher, generate_key
from gsn_receiver import GSNReceiver
from gsn_key_matcher import LiveKDRPlotter


# ---------------- Key State ----------------
key_lock = threading.Lock()
gsn_raw = None
keys_by_epoch = {}   # epoch -> aes_key


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
    global gsn_raw
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


# ---------------- RS Receiver Thread ----------------
def rs_thread(plotter):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5007))
    epoch = 0

    while True:
        data, _ = sock.recvfrom(65535)
        parts = data.decode().split()
        if len(parts) != 4:
            continue

        epoch   = int(parts[1])
        uav_raw = parts[2]
        parity  = parts[3]
       
        with key_lock:
            local_raw = gsn_raw
        if local_raw is None:
            continue

        corrected = rs_decode_key(local_raw, parity)
        aes = sha256.sha_byte(corrected)

        with key_lock:
            keys_by_epoch[epoch] = aes
            print(
                f"[KEY ACTIVE] epoch={epoch} | AES={aes.hex()[:32]}..."
            )

        

        plotter.update(uav_raw, local_raw, corrected)
        epoch += 1


# ---------------- Main ----------------
if __name__ == "__main__":
    # 1️⃣ 建立 plotter（但不要啟動）
    plotter = LiveKDRPlotter()

    # 2️⃣ 啟動 background threads（key / RS）
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
