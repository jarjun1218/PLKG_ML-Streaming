import argparse
import glob
import os
import threading
import time
from datetime import datetime

import serial

# ============ path config ============
Data_Path = r"data_collecting_processing/CSI_data/processed_with_ros/"
os.makedirs(Data_Path, exist_ok=True)
Defult = r"csi_data"
filetype = r".txt"

BOOT_PATTERNS = (
    "rst:",
    "boot:",
    "ets ",
    "load:",
    "entry ",
    "brownout",
    "guru meditation",
    "panic",
    "abort()",
    "rebooting",
)


def connect_serial(port, baudrate):
    """Open serial port with conservative settings for long-running CSI reads."""
    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=1,
        write_timeout=1,
        inter_byte_timeout=0.1,
        exclusive=False,
    )

    # Avoid consuming stale bytes left in the adapter / device buffer.
    time.sleep(0.2)
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    return ser


def generate_filename(endpoint_type):
    """
    data_collecting_processing/CSI_data/processed_with_ros/<endpoint_type>/csi_dataX.txt
    """
    base_dir = os.path.join(Data_Path, endpoint_type)
    os.makedirs(base_dir, exist_ok=True)

    pattern = os.path.join(base_dir, f"{Defult}*{filetype}")
    index = 1
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = stem[len(Defult):]
        if suffix.isdigit():
            index = max(index, int(suffix) + 1)

    return os.path.join(base_dir, f"{Defult}{index}{filetype}")


class CSISerialStreamer:
    """
    Robust serial CSI reader with reboot/noise filtering.

    - Ignores non-CSI boot log lines from ESP32 resets
    - Drops duplicated / out-of-order serial_num data
    - Flushes buffers after reconnect or reboot detection
    - Reconnects automatically on SerialException
    """

    def __init__(
        self,
        port,
        baudrate,
        endpoint_type,
        write_file=True,
        callback=None,
        reboot_cooldown=1.5,
        reconnect_delay=1.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.endpoint_type = endpoint_type
        self.write_file = write_file
        self.callback = callback
        self.reboot_cooldown = reboot_cooldown
        self.reconnect_delay = reconnect_delay

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ser = None
        self._file = None
        self._last_serial = None
        self._discard_until = 0.0

    def start(self):
        self._stop_event.clear()
        self._ensure_serial()

        if self.write_file and self._file is None:
            file_path = generate_filename(self.endpoint_type)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._file = open(file_path, "a", encoding="utf-8")
            print(f"[CSI] Logging to {file_path}")

        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        self._close_serial()

        if self._file:
            self._file.close()
            self._file = None

    def _ensure_serial(self):
        if self._ser is not None and self._ser.is_open:
            return
        self._ser = connect_serial(self.port, self.baudrate)
        self._last_serial = None
        self._discard_until = time.time() + self.reboot_cooldown
        print(f"[CSI] Serial connected: {self.port} @ {self.baudrate}")

    def _close_serial(self):
        ser = self._ser
        self._ser = None
        if not ser:
            return
        try:
            if ser.is_open:
                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception:
                    pass
                ser.close()
        except Exception:
            pass

    def _trigger_reboot_recovery(self, reason):
        print(f"[CSI] Reset/noise detected on {self.port}: {reason}")
        self._last_serial = None
        self._discard_until = time.time() + self.reboot_cooldown
        try:
            if self._ser and self._ser.is_open:
                self._ser.reset_input_buffer()
        except Exception:
            pass

    @staticmethod
    def _looks_like_boot_log(line):
        lower = line.lower()
        return any(token in lower for token in BOOT_PATTERNS)

    @staticmethod
    def _extract_serial_num(line):
        if "serial_num:" not in line:
            return None
        try:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            idx = parts.index("serial_num:")
            return int(parts[idx + 1])
        except Exception:
            return None

    def _accept_line(self, line):
        now = time.time()

        if not line:
            return False

        if self._looks_like_boot_log(line):
            self._trigger_reboot_recovery(line[:80])
            return False

        if now < self._discard_until:
            return False

        serial_num = self._extract_serial_num(line)
        if serial_num is None:
            return False

        if self._last_serial is None:
            self._last_serial = serial_num
            return True

        if serial_num == self._last_serial:
            return False

        if serial_num < self._last_serial:
            # ESP32 reboot or buffered old data often makes serial_num jump backwards.
            if self._last_serial - serial_num > 5:
                self._trigger_reboot_recovery(
                    f"serial rolled back from {self._last_serial} to {serial_num}"
                )
            return False

        self._last_serial = serial_num
        return True

    def _run(self):
        print(f"[CSI] Start collecting on {self.port} ({self.endpoint_type})")
        if self._file:
            print(f"[CSI] Writing CSI data in {self._file.name}")

        while not self._stop_event.is_set():
            try:
                self._ensure_serial()
                raw = self._ser.readline()
                if not raw:
                    continue

                line = raw.decode(errors="ignore").strip()
                if not self._accept_line(line):
                    continue

                stamped = f"{datetime.now()} {line}"

                if self._file:
                    self._file.write(stamped + "\n\n")
                    self._file.flush()

                if self.callback:
                    try:
                        self.callback(line)
                    except Exception as exc:
                        print(f"[CSI] callback error: {exc}")

            except (serial.SerialException, OSError) as exc:
                print(f"[CSI] Serial error on {self.port}: {exc}")
                self._close_serial()
                if self._stop_event.is_set():
                    break
                time.sleep(self.reconnect_delay)
            except Exception as exc:
                print(f"[CSI] Stream error: {exc}")
                time.sleep(0.1)

        self._close_serial()
        print("[CSI] Collector stopped.")


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
