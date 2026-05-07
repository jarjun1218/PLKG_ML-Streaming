import serial
import threading
import numpy as np
from datetime import datetime
import platform

NO_USED_CARRIERS_CH1 = [0, 1, 2, 3, 4, 5, 11, 32, 59, 60, 61, 62, 63]
NO_USED_CARRIERS_CH13 = [0, 1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
EVE_NO_USED_CARRIERS = [0, 1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
CSI_NORMALIZE_DENOMINATOR = 61.846


def _line_payload(fields, start):
    if len(fields) <= start:
        return []
    if fields[-1] == "":
        return fields[start:-1]
    return fields[start:]


def _preprocess_amp(csi_amp):
    if len(csi_amp) == 64:
        if csi_amp[2] == 0:
            return np.delete(csi_amp, NO_USED_CARRIERS_CH1)
        return np.delete(csi_amp, NO_USED_CARRIERS_CH13)
    if len(csi_amp) == 128:
        csi_amp = csi_amp.reshape(64, 2)
        csi_amp = np.sqrt(csi_amp[:, 0] ** 2 + csi_amp[:, 1] ** 2)
        if csi_amp[2] == 0:
            return np.delete(csi_amp, NO_USED_CARRIERS_CH1)
        return np.delete(csi_amp, NO_USED_CARRIERS_CH13)
    return None


def preprocess_csi_line(line):
    """
    Shared ESP32 CSI preprocessing used by the live UAV/GSN key pipeline.
    Mirrors CSIReader's field layout, carrier deletion, and fixed normalization.
    """
    try:
        if "serial_num:" in line:
            data = line.split(",")[1:]
            if len(data) < 6:
                return None

            serial_num = int(data[0])
            csi_type = data[1]
            rssi = float(data[2])
            noise = float(data[3])

            if csi_type in ("384", "256"):
                device = "GCS"
            elif csi_type == "128":
                device = "UAV"
            else:
                return None

            csi_amp = np.array(_line_payload(data, 5), dtype=np.float32)
            raw_csi = _preprocess_amp(csi_amp)
            if raw_csi is None or len(raw_csi) != 51:
                return None

            csi = (raw_csi - 0) / (CSI_NORMALIZE_DENOMINATOR - 0)
            csi = csi.astype(np.float32)
            return {
                "serial": serial_num,
                "device": device,
                "rssi": rssi,
                "noise": noise,
                "raw_csi": raw_csi.astype(np.float32),
                "csi": csi,
                "combined": np.concatenate(([serial_num, rssi, noise], csi)).astype(np.float32),
            }

        if "eve," in line:
            data = line.split(",")[1:]
            if len(data) < 6:
                return None

            rssi = float(data[2])
            noise = float(data[3])
            mac = data[4]
            csi_amp = np.array(_line_payload(data, 5), dtype=np.float32)
            raw_csi = np.delete(csi_amp, EVE_NO_USED_CARRIERS)
            if len(raw_csi) != 51:
                return None

            csi = (raw_csi - 0) / (CSI_NORMALIZE_DENOMINATOR - 0)
            csi = csi.astype(np.float32)
            return {
                "serial": 0,
                "device": "EVE",
                "mac": mac,
                "rssi": rssi,
                "noise": noise,
                "raw_csi": raw_csi.astype(np.float32),
                "csi": csi,
                "combined": np.concatenate(([0, rssi, noise], csi)).astype(np.float32),
            }
    except Exception:
        return None

    return None

# CSI reader module for ESP32
class CSIReader:
    def __init__(self, port='', baud=115200, timeout=0.1):
        # Initialize the serial port
        self.ser = serial.Serial(port, baud, timeout=timeout)

        # Clear the input and output buffers
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        # Set a lock for reading the serial data
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        
        # Initialize variables
        self.serial_num = 0
        self.timestamp = datetime.now().strftime("%H:%M:%S") + '.%02d' % (datetime.now().microsecond // 10000)
        self.rssi = 0
        self.noise = 0
        self.mac = ""
        self.device = ""
        self.raw_csi = np.zeros(51, dtype=np.float32)
        self.csi_amp_array = np.zeros(51, dtype=np.float32)
        self.csi_amp_filtered = np.zeros(51, dtype=np.float32)
        self.combined = np.zeros(54, dtype=np.float32)

    def start(self):
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread.start()
            
    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
        if self.ser.is_open:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.ser.close()
            print("Serial port closed.")
        
    def _monitor(self):
        try:
            while not self._stop.is_set():
                if not self.ser.is_open:
                    break
                # Read a line from the serial port
                line = self.ser.readline().decode(errors='ignore').strip()  # Read a line from the serial port
                self.timestamp = datetime.now().strftime("%H:%M:%S") + '.%02d' % (datetime.now().microsecond // 10000)  # Get current timestamp
                sample = preprocess_csi_line(line)
                if sample:
                    with self._lock:
                        self.serial_num = sample["serial"]
                        self.rssi = sample["rssi"]
                        self.noise = sample["noise"]
                        self.device = sample["device"]
                        self.mac = sample.get("mac", "")
                        self.raw_csi = sample["raw_csi"].copy()
                        self.csi_amp_array = sample["csi"].copy()
                        self.combined = sample["combined"].copy()
                else:
                    pass
        except serial.SerialException as e:
            print(f"{self.device} Serial error: {e}")
        except Exception as e:
            print(f"{self.device} Error: {e}")
            pass
        finally:
            # Ensure the serial port is closed properly
            if self.ser.is_open:
                self.ser.close()
                print(f"{self.device} Serial port closed.")
            else:
                print(f"{self.device} Serial port already closed.")
    
    # real-time plot using matplotlib animation
    def start_plot(self, interval=100):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        self.fig, self.ax = plt.subplots()
        x = np.arange(self.csi_amp_array.size)
        self.line, = self.ax.plot(x, self.csi_amp_array)
        self.ax.set_xlim(0, self.csi_amp_array.size - 1)
        self.ax.set_ylim(0, 100)
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Amplitude')
        plt.title('CSI Amplitude')
        plt.grid()
        
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update_plot, 
            interval=interval,
            cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()
        
    def update_plot(self, frame):
        with self._lock:
            data = self.csi_amp_array.copy()
        if data.size != self.line.get_ydata().size:
            # print("Data size mismatch, skipping update.")
            return (self.line,)
        else:
            self.line.set_ydata(data)
            self.ax.set_ylim(0.1, max(0.5, data.max() * 1.2))  # Adjust y-limits dynamically
            return (self.line,)

    def stop_plot(self):
        if self.ani:
            # plt.close(self.fig)
            print("Plot closed.")
        else:
            print("No plot to close.")
                
if __name__ == "__main__":
    if platform.system() == "Darwin":  # MacOS
        reader = CSIReader(port='/dev/cu.usbserial-0001', baud=115200)
    elif platform.system() == "Windows":
        reader = CSIReader(port='COM3', baud=115200)
    elif platform.system() == "Linux":
        reader = CSIReader(port='/dev/ttyUSB0', baud=115200)
    else:
        raise EnvironmentError("Unsupported platform")
    reader.start()
    try:
        reader.start_plot(interval=100)  # Start the plot with a 100ms interval
    except KeyboardInterrupt:
        reader.stop()
    finally:
        reader.stop()
        print("Program terminated.")
        reader.stop_plot()
        
        
