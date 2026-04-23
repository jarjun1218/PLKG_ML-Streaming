import argparse
import glob
import threading
import time
from datetime import datetime
import os

import serial

# ============ 路徑與檔名設定 ============
Data_Path = r"data_collecting_processing/CSI_data/processed_with_ros/"
os.makedirs(Data_Path, exist_ok=True)
Defult = r"csi_data"
filetype = r".txt"


def connect_serial(port, baudrate):
    """跟舊版一樣的開 serial 寫法。"""
    timeout = 1  # set timeout
    ser = serial.Serial(port, baudrate, timeout=timeout)
    return ser


def generate_filename(endpoint_type):
    """
    使用較穩定的 glob+自動遞增邏輯：
    data_collecting_processing/CSI_data/processed_with_ros/<endpoint_type>/csi_dataX.txt
    """
    base_dir = os.path.join(Data_Path, endpoint_type)
    os.makedirs(base_dir, exist_ok=True)

    pattern = os.path.join(base_dir, f"{Defult}*{filetype}")
    index = 1
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]  # csi_data12
        suffix = stem[len(Defult):]  # "12"
        if suffix.isdigit():
            index = max(index, int(suffix) + 1)

    return os.path.join(base_dir, f"{Defult}{index}{filetype}")


class CSISerialStreamer:
    """
    用「舊版 collect.py 的 read_csi() 行為」做成的長時間 CSI reader。

    - 會用 ser.readline().decode().strip()
    - 只處理含 "serial_num:" 的行
    - 每行前面加 datetime.now()
    - 可以同時寫檔 + 丟給 callback（給 live_key_stream 用）
    """

    def __init__(self, port, baudrate, endpoint_type, write_file=True, callback=None):
        self.port = port
        self.baudrate = baudrate
        self.endpoint_type = endpoint_type
        self.write_file = write_file
        self.callback = callback

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ser = None
        self._file = None

    def start(self):
        self._stop_event.clear()

        # 開 serial（跟舊版 connect_serial 一樣）
        if self._ser is None or not self._ser.is_open:
            self._ser = connect_serial(self.port, self.baudrate)

        # 開檔案
        if self.write_file and self._file is None:
            file_path = generate_filename(self.endpoint_type)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._file = open(file_path, "a")
            print(f"[CSI] Logging to {file_path}")

        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)

        if self._ser and self._ser.is_open:
            self._ser.close()
            self._ser = None
        # clear buffer
        if self._ser:
            self._ser.flushInput()
            self._ser.flushOutput()

        if self._file:
            self._file.close()
            self._file = None

    def _run(self):
        print(f"[CSI] Start collecting on {self.port} ({self.endpoint_type})")
        if self._file:
            print(f"[CSI] Writing CSI data in {self._file.name}")

        try:
            while not self._stop_event.is_set():
                # ★ 完全沿用舊版：readline → decode → strip
                line = self._ser.readline().decode(errors="ignore").strip()

                # 舊版就是只處理有 serial_num 的行
                if "serial_num:" not in line:
                    continue

                stamped = f"{datetime.now()} {line}"

                # 寫檔（加兩個換行，跟舊版一樣）
                if self._file:
                    self._file.write(stamped + "\n\n")
                    self._file.flush()

                # 丟給 callback 給 live_key_stream 用
                if self.callback:
                    try:
                        self.callback(line)
                    except Exception as exc:
                        print(f"[CSI] callback error: {exc}")

        except Exception as exc:
            print(f"[CSI] Stream stopped due to error: {exc}")
        finally:
            print("[CSI] Collector stopped.")


# ============ CLI（新版 collect 主程式）============
def build_argparser():
    parser = argparse.ArgumentParser(description="Continuous CSI collector")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port to read from")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--endpoint-type", default="GSN", help="Endpoint tag for file naming")
    parser.add_argument("--disable-file", action="store_true", help="Do not write raw data to disk")
    return parser


def main():
    args = build_argparser().parse_args()
    streamer = CSISerialStreamer(
        args.port,
        args.baudrate,
        args.endpoint_type,
        write_file=not args.disable_file,
    )
    streamer.start()
    print("[CSI] Press Ctrl+C to stop collecting.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
