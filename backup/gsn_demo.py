import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import threading
import time
import socket
import queue
from collections import deque
from dataclasses import dataclass, field

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sha256
from csi_control import send_csi_reset_request
from key_confirm import verify_key_confirm
from rs_ecc import rs_decode_key
from gsn_key_generate import load_model, CSISerialWatcher, generate_key
from gsn_receiver import GSNReceiver

UAV_CONTROL_PORT = 5008
RAW_HISTORY_LIMIT = 512


@dataclass
class GSNState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    model_loaded: bool = False
    serial_connected: bool = False
    receiver_started: bool = False
    rs_started: bool = False

    gsn_raw: str | None = None
    gsn_raw_by_serial: dict = field(default_factory=dict)
    latest_serial: int | None = None
    latest_rssi: float | None = None
    latest_noise: float | None = None
    latest_csi_time: float | None = None
    pending_uav_csi_reset: bool = True

    last_epoch: int | None = None
    active_key: bytes | None = None
    keys_by_epoch: dict = field(default_factory=dict)

    latest_latency_ms: float | None = None
    latest_latency_ema_ms: float | None = None
    latest_frame_bgr: np.ndarray | None = None
    latest_frame_time: float | None = None
    video_status: str = "Waiting for frames..."
    video_status_level: str = "idle"

    latest_kdr_raw: float | None = None
    latest_kdr_corr: float | None = None
    kdr_raw_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    kdr_corr_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    latency_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    latency_ema_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    rssi_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    hist_idx: int = 0


class GSNDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GSN Dashboard - PLKG Ground Station")
        self.geometry("1366x768")
        self.minsize(800, 500)
        self.configure(bg="#0f172a")

        self.state_obj = GSNState()
        self.log_queue = queue.Queue()
        self.after_ids = []
        self.ui_paused = False
        self.backend_threads = []
        self.video_photo = None

        self._configure_style()
        self._build_layout()
        self._schedule_updates()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Root.TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#111827", relief="flat")
        style.configure("Title.TLabel", background="#111827", foreground="#e5e7eb", font=("Arial", 12, "bold"))
        style.configure("Value.TLabel", background="#111827", foreground="#f8fafc", font=("Consolas", 11, "bold"))
        style.configure("Muted.TLabel", background="#111827", foreground="#94a3b8", font=("Arial", 10))
        style.configure("Header.TLabel", background="#0f172a", foreground="#f8fafc", font=("Arial", 15, "bold"))
        style.configure("TButton", font=("Arial", 11, "bold"), padding=8)
        style.configure("StatusIdle.TLabel", background="#111827", foreground="#94a3b8", font=("Arial", 10, "bold"))
        style.configure("StatusWarn.TLabel", background="#111827", foreground="#fbbf24", font=("Arial", 10, "bold"))
        style.configure("StatusGood.TLabel", background="#111827", foreground="#34d399", font=("Arial", 10, "bold"))
        style.configure("StatusBad.TLabel", background="#111827", foreground="#f87171", font=("Arial", 10, "bold"))

    def _build_layout(self):
        root = ttk.Frame(self, style="Root.TFrame", padding=12)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root, style="Root.TFrame")
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="GSN PLKG Dashboard", style="Header.TLabel").pack(side="left")
        self.status_banner = ttk.Label(top, text="Backend not started", style="Muted.TLabel")
        self.status_banner.pack(side="right")

        control = ttk.Frame(root, style="Card.TFrame", padding=10)
        control.pack(fill="x", pady=(0, 10))
        self.start_btn = ttk.Button(control, text="Start GSN Backend", command=self.start_backend)
        self.start_btn.pack(side="left", padx=(0, 8))
        self.pause_btn = ttk.Button(control, text="Pause UI Refresh", command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=(0, 8))
        self.clear_btn = ttk.Button(control, text="Clear Log", command=self.clear_log)
        self.clear_btn.pack(side="left")
        ttk.Label(control, text="Stop does not terminate backend threads in this version.", style="Muted.TLabel").pack(side="right")

        stats = ttk.Frame(root, style="Root.TFrame")
        stats.pack(fill="x", pady=(0, 10))
        stats.columnconfigure((0,1,2,3,4,5), weight=1)

        self.cards = {}
        card_specs = [
            ("serial", "Latest Serial"),
            ("rssi", "RSSI"),
            ("noise", "Noise"),
            ("epoch", "Active Epoch"),
            ("latency", "Latency (ms)"),
            ("kdr", "Correction Bits"),
        ]
        for idx, (key, title) in enumerate(card_specs):
            frame = ttk.Frame(stats, style="Card.TFrame", padding=10)
            frame.grid(row=0, column=idx, sticky="nsew", padx=4)
            ttk.Label(frame, text=title, style="Title.TLabel").pack(anchor="w")
            value = ttk.Label(frame, text="--", style="Value.TLabel")
            value.pack(anchor="w", pady=(8, 0))
            self.cards[key] = value

        body = ttk.Frame(root, style="Root.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=3)
        body.rowconfigure(1, weight=2)

        video_card = ttk.Frame(body, style="Card.TFrame", padding=10)
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        ttk.Label(video_card, text="Decrypted Video", style="Title.TLabel").pack(anchor="w")
        self.video_label = tk.Label(video_card, text="Waiting for frames...", bg="#020617", fg="#cbd5e1")
        self.video_label.pack(fill="both", expand=True, pady=(8, 0))
        self.video_hint = ttk.Label(video_card, text="Waiting for UAV video stream.", style="StatusIdle.TLabel")
        self.video_hint.pack(anchor="w", pady=(8, 0))

        right = ttk.Frame(body, style="Root.TFrame")
        right.grid(row=0, column=1, sticky="nsew", pady=(0, 6))
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        key_card = ttk.Frame(right, style="Card.TFrame", padding=10)
        key_card.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        ttk.Label(key_card, text="Key Status", style="Title.TLabel").pack(anchor="w")
        self.key_text = tk.Text(key_card, height=7, wrap="word", bg="#020617", fg="#e2e8f0", insertbackground="#e2e8f0", relief="flat")
        self.key_text.pack(fill="both", expand=True, pady=(8, 0))
        self.key_text.insert("end", "GSN key state will appear here.\n")
        self.key_text.config(state="disabled")

        hist_card = ttk.Frame(right, style="Card.TFrame", padding=10)
        hist_card.grid(row=1, column=0, sticky="nsew")
        ttk.Label(hist_card, text="Epoch History", style="Title.TLabel").pack(anchor="w")
        self.epoch_list = tk.Text(hist_card, height=7, wrap="none", bg="#020617", fg="#dbeafe", relief="flat")
        self.epoch_list.pack(fill="both", expand=True, pady=(8, 0))
        self.epoch_list.insert("end", "No epochs yet.\n")
        self.epoch_list.config(state="disabled")

        charts = ttk.Frame(body, style="Root.TFrame")
        charts.grid(row=1, column=0, columnspan=2, sticky="nsew")
        charts.columnconfigure(0, weight=1)
        charts.columnconfigure(1, weight=1)
        charts.rowconfigure(0, weight=1)

        self._build_chart_card(charts, 0, "Correction Trend", "kdr")
        self._build_chart_card(charts, 1, "Latency", "lat")

        log_card = ttk.Frame(root, style="Card.TFrame", padding=10)
        log_card.pack(fill="both", expand=False, pady=(10, 0))
        ttk.Label(log_card, text="System Log", style="Title.TLabel").pack(anchor="w")
        self.log_text = tk.Text(log_card, height=6, wrap="word", bg="#020617", fg="#e2e8f0", insertbackground="#e2e8f0", relief="flat")
        self.log_text.pack(fill="both", expand=True, pady=(8, 0))

    def _build_chart_card(self, parent, column, title, kind):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        frame.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0))
        ttk.Label(frame, text=title, style="Title.TLabel").pack(anchor="w")
        fig = Figure(figsize=(4.2, 2.1), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#020617")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        if kind == "kdr":
            self.kdr_fig, self.kdr_ax, self.kdr_canvas = fig, ax, canvas
        else:
            self.lat_fig, self.lat_ax, self.lat_canvas = fig, ax, canvas

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}] {message}")

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def toggle_pause(self):
        self.ui_paused = not self.ui_paused
        self.pause_btn.config(text="Resume UI Refresh" if self.ui_paused else "Pause UI Refresh")
        self.status_banner.config(text="UI paused" if self.ui_paused else "UI running")

    def start_backend(self):
        with self.state_obj.lock:
            if self.state_obj.started:
                self.log("Backend already started.")
                return
            self.state_obj.started = True
        self.start_btn.state(["disabled"])
        self.status_banner.config(text="Starting backend...")
        self.log("Starting GSN backend.")
        try:
            send_csi_reset_request()
            self.log("Sent RESET_CSI request to UAV.")
        except Exception as e:
            self.log(f"Failed to send RESET_CSI request: {e}")
        with self.state_obj.lock:
            self.state_obj.video_status = "Backend starting. Waiting for UAV key and video."
            self.state_obj.video_status_level = "idle"
            self.state_obj.pending_uav_csi_reset = True

        t1 = threading.Thread(target=self._keygen_worker, daemon=True)
        t2 = threading.Thread(target=self._rs_worker, daemon=True)
        t1.start()
        t2.start()
        self.backend_threads.extend([t1, t2])

        try:
            rx = GSNReceiver(get_aes_key=self._get_key, on_frame=self._handle_frame)
            rx.start()
            with self.state_obj.lock:
                self.state_obj.receiver_started = True
            self.log("GSNReceiver started on UDP/5005.")
        except Exception as e:
            self.log(f"Failed to start GSNReceiver: {e}")

    def _keygen_worker(self):
        try:
            model = load_model("model_reserved/cnn_basic/model_final.pth")
            # watcher = CSISerialWatcher("/dev/ttyUSB0", 115200)
            # watcher = CSISerialWatcher("/dev/cu.usbserial-0001", 115200)  # macOS
            watcher = CSISerialWatcher("COM3", 115200)  # Windows
            watcher.start()
            with self.state_obj.lock:
                self.state_obj.model_loaded = True
                self.state_obj.serial_connected = True
            self.log(f"CSI model loaded. Serial watcher started on.")
        except Exception as e:
            self.log(f"Keygen init failed: {e}")
            return

        last_serial = None
        while True:
            try:
                snap = watcher.snapshot().get("GSN")
                if not snap:
                    time.sleep(0.01)
                    continue
                serial = snap.get("serial")
                csi = snap.get("csi")
                rssi = snap.get("rssi")
                noise = snap.get("noise")
                csi_time = snap.get("time")

                if serial == last_serial:
                    time.sleep(0.002)
                    continue
                last_serial = serial

                if csi is None:
                    continue
                csi = np.asarray(csi, dtype=np.float32)
                if csi.ndim != 1 or len(csi) < 10:
                    continue

                raw, _ = generate_key(model, csi)
                with self.state_obj.lock:
                    self.state_obj.gsn_raw = raw
                    self.state_obj.gsn_raw_by_serial[serial] = raw
                    if len(self.state_obj.gsn_raw_by_serial) > RAW_HISTORY_LIMIT:
                        overflow = len(self.state_obj.gsn_raw_by_serial) - RAW_HISTORY_LIMIT
                        oldest = sorted(self.state_obj.gsn_raw_by_serial)[:overflow]
                        for old_serial in oldest:
                            self.state_obj.gsn_raw_by_serial.pop(old_serial, None)
                    self.state_obj.latest_serial = serial
                    self.state_obj.latest_rssi = rssi
                    self.state_obj.latest_noise = noise
                    self.state_obj.latest_csi_time = csi_time
                    self.state_obj.rssi_hist.append(rssi)
                if serial % 20 == 0:
                    self.log(f"Generated local raw key from serial {serial}.")
            except Exception as e:
                self.log(f"Keygen loop error: {e}")
                time.sleep(0.1)

    def _rs_worker(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            sock.bind(("0.0.0.0", 5007))
            with self.state_obj.lock:
                self.state_obj.rs_started = True
            self.log("RS parity receiver started on UDP/5007.")
        except Exception as e:
            self.log(f"RS receiver init failed: {e}")
            return

        last_epoch = -1
        rejected_epochs = set()
        while True:
            try:
                data, addr = sock.recvfrom(1024*1024)
                parts = data.decode(errors="ignore").split()
                if len(parts) != 5 or parts[0] != "R":
                    continue
                epoch = int(parts[1])
                serial = int(parts[2])
                parity = parts[3]
                confirm = parts[4]

                with self.state_obj.lock:
                    need_reset = self.state_obj.pending_uav_csi_reset
                    if need_reset:
                        self.state_obj.pending_uav_csi_reset = False

                if need_reset:
                    self._request_uav_csi_reset(addr[0])
                    with self.state_obj.lock:
                        self.state_obj.keys_by_epoch.clear()
                        self.state_obj.last_epoch = None
                        self.state_obj.active_key = None
                        self.state_obj.latest_frame_bgr = None
                        self.state_obj.latest_frame_time = None
                        self.state_obj.video_status = "Requested UAV CSI reset. Waiting for fresh epoch."
                        self.state_obj.video_status_level = "warn"
                    self.log(f"Sent RESET_CSI to UAV at {addr[0]}")
                    last_epoch = -1
                    rejected_epochs.clear()
                    continue

                if last_epoch >= 0 and epoch < last_epoch:
                    with self.state_obj.lock:
                        self.state_obj.keys_by_epoch.clear()
                        self.state_obj.last_epoch = None
                        self.state_obj.active_key = None
                        self.state_obj.latest_frame_bgr = None
                        self.state_obj.latest_frame_time = None
                        self.state_obj.video_status = "UAV reboot detected. Waiting for key resync and fresh video."
                        self.state_obj.video_status_level = "warn"
                    self.log(f"UAV key session reset detected: epoch {last_epoch}->{epoch}")
                    rejected_epochs.clear()
                last_epoch = epoch

                with self.state_obj.lock:
                    if epoch in self.state_obj.keys_by_epoch:
                        continue
                if epoch in rejected_epochs:
                    continue

                with self.state_obj.lock:
                    local_raw = self.state_obj.gsn_raw_by_serial.get(serial)
                if local_raw is None:
                    self.log(f"Waiting for local CSI serial={serial} for epoch={epoch}.")
                    continue

                try:
                    corrected = rs_decode_key(local_raw, parity)
                except ValueError as e:
                    self.log(f"RS correction failed for epoch={epoch}, serial={serial}: {e}")
                    continue
                aes = sha256.sha_byte(corrected)
                if not verify_key_confirm(aes, epoch, serial, parity, confirm):
                    rejected_epochs.add(epoch)
                    with self.state_obj.lock:
                        self.state_obj.video_status = "Key confirmation failed. Waiting for next epoch."
                        self.state_obj.video_status_level = "warn"
                        self.state_obj.last_epoch = None
                        self.state_obj.active_key = None
                    self.log(f"Key confirmation failed for epoch={epoch}, serial={serial}.")
                    continue

                raw_kdr = self._kdr(local_raw, corrected) * 100.0

                with self.state_obj.lock:
                    self.state_obj.keys_by_epoch[epoch] = aes
                    self.state_obj.last_epoch = epoch
                    self.state_obj.active_key = aes
                    frame_is_stale = (
                        self.state_obj.latest_frame_time is None
                        or (time.time() - self.state_obj.latest_frame_time) > 1.0
                    )
                    if frame_is_stale:
                        self.state_obj.video_status = "Key synced. Waiting for fresh decrypted video."
                        self.state_obj.video_status_level = "warn"
                    self.state_obj.latest_kdr_raw = raw_kdr
                    self.state_obj.latest_kdr_corr = None
                    self.state_obj.kdr_raw_hist.append(raw_kdr)
                    self.state_obj.hist_idx += 1
                self.log(f"[KEY ACTIVE] epoch={epoch} serial={serial} confirmed corrected_bits={raw_kdr:.2f}%")
            except Exception as e:
                self.log(f"RS loop error: {e}")
                time.sleep(0.05)

    @staticmethod
    def _request_uav_csi_reset(uav_ip):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for _ in range(3):
                sock.sendto(b"RESET_CSI", (uav_ip, UAV_CONTROL_PORT))
                time.sleep(0.05)
        finally:
            sock.close()

    def _handle_frame(self, frame, latency):
        if frame is None:
            return
        with self.state_obj.lock:
            latency_value = float(latency)
            prev_ema = self.state_obj.latest_latency_ema_ms
            ema = latency_value if prev_ema is None else (prev_ema * 0.8 + latency_value * 0.2)
            self.state_obj.latest_frame_bgr = frame.copy()
            self.state_obj.latest_latency_ms = latency_value
            self.state_obj.latest_latency_ema_ms = ema
            self.state_obj.latest_frame_time = time.time()
            self.state_obj.video_status = "Video streaming normally."
            self.state_obj.video_status_level = "good"
            self.state_obj.latency_hist.append(latency_value)
            self.state_obj.latency_ema_hist.append(ema)

    def _get_key(self, epoch):
        with self.state_obj.lock:
            return self.state_obj.keys_by_epoch.get(epoch)

    @staticmethod
    def _kdr(a, b):
        if not a or not b:
            return 0.0
        L = min(len(a), len(b))
        if L == 0:
            return 0.0
        return sum(a[i] != b[i] for i in range(L)) / L

    def _schedule_updates(self):
        self.after_ids.append(self.after(50, self._drain_log_queue))
        self.after_ids.append(self.after(10, self._refresh_ui))

    def _drain_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
        except queue.Empty:
            pass
        self.after_ids.append(self.after(200, self._drain_log_queue))

    def _refresh_ui(self):
        if not self.ui_paused:
            with self.state_obj.lock:
                serial = self.state_obj.latest_serial
                rssi = self.state_obj.latest_rssi
                noise = self.state_obj.latest_noise
                epoch = self.state_obj.last_epoch
                latency = self.state_obj.latest_latency_ms
                latency_ema = self.state_obj.latest_latency_ema_ms
                frame_time = self.state_obj.latest_frame_time
                raw_kdr = self.state_obj.latest_kdr_raw
                corr_kdr = self.state_obj.latest_kdr_corr
                frame = None if self.state_obj.latest_frame_bgr is None else self.state_obj.latest_frame_bgr.copy()
                gsn_raw = self.state_obj.gsn_raw
                aes_key = self.state_obj.active_key
                video_status = self.state_obj.video_status
                video_status_level = self.state_obj.video_status_level
                keys_by_epoch = dict(self.state_obj.keys_by_epoch)
                kdr_raw_hist = list(self.state_obj.kdr_raw_hist)
                kdr_corr_hist = list(self.state_obj.kdr_corr_hist)
                lat_hist = list(self.state_obj.latency_hist)
                lat_ema_hist = list(self.state_obj.latency_ema_hist)
                rssi_hist = list(self.state_obj.rssi_hist)

            self.cards["serial"].config(text="--" if serial is None else str(serial))
            self.cards["rssi"].config(text="--" if rssi is None else f"{rssi:.1f}")
            self.cards["noise"].config(text="--" if noise is None else f"{noise:.1f}")
            self.cards["epoch"].config(text="--" if epoch is None else str(epoch))
            if latency is None:
                self.cards["latency"].config(text="--")
            elif latency_ema is None:
                self.cards["latency"].config(text=f"{latency:.1f}")
            else:
                self.cards["latency"].config(text=f"{latency_ema:.1f} avg")
            if raw_kdr is None:
                self.cards["kdr"].config(text="--")
            elif corr_kdr is None:
                self.cards["kdr"].config(text=f"{raw_kdr:.2f}%")
            else:
                self.cards["kdr"].config(text=f"{raw_kdr:.2f}% / {corr_kdr:.2f}%")

            self._update_key_panel(serial, rssi, noise, gsn_raw, epoch, aes_key)
            self._update_epoch_history(keys_by_epoch)
            self._update_video(frame, latency_ema, frame_time, aes_key, video_status, video_status_level)
            self._update_kdr_chart(kdr_raw_hist, kdr_corr_hist)
            self._update_latency_chart(lat_hist, lat_ema_hist)

            with self.state_obj.lock:
                running = self.state_obj.started
                model_loaded = self.state_obj.model_loaded
                serial_ok = self.state_obj.serial_connected
                rx_ok = self.state_obj.receiver_started
                rs_ok = self.state_obj.rs_started
            self.status_banner.config(
                text=f"backend={'ON' if running else 'OFF'} | model={'ON' if model_loaded else 'OFF'} | serial={'ON' if serial_ok else 'OFF'} | videoRX={'ON' if rx_ok else 'OFF'} | rsRX={'ON' if rs_ok else 'OFF'}"
            )

        self.after_ids.append(self.after(33, self._refresh_ui))

    def _update_key_panel(self, serial, rssi, noise, gsn_raw, epoch, aes_key):
        lines = [
            f"serial      : {serial if serial is not None else '--'}",
            f"rssi/noise  : {('--' if rssi is None else f'{rssi:.1f}')}/{('--' if noise is None else f'{noise:.1f}')}",
            f"raw key     : {(gsn_raw[:64] + '...') if gsn_raw else '--'}",
            f"active epoch : {epoch if epoch is not None else '--'}",
            f"aes key     : {(aes_key.hex()[:48] + '...') if isinstance(aes_key, (bytes, bytearray)) else '--'}",
        ]
        self.key_text.config(state="normal")
        self.key_text.delete("1.0", "end")
        self.key_text.insert("end", "\n".join(lines))
        self.key_text.config(state="disabled")

    def _update_epoch_history(self, keys_by_epoch):
        lines = []
        for epoch in sorted(keys_by_epoch.keys(), reverse=True)[:20]:
            aes = keys_by_epoch[epoch]
            key_str = aes.hex()[:32] + "..." if isinstance(aes, (bytes, bytearray)) else str(aes)
            lines.append(f"epoch {epoch:<6} key {key_str}")
        if not lines:
            lines = ["No epochs yet."]
        self.epoch_list.config(state="normal")
        self.epoch_list.delete("1.0", "end")
        self.epoch_list.insert("end", "\n".join(lines))
        self.epoch_list.config(state="disabled")

    def _update_video(self, frame, latency, frame_time, aes_key, video_status, video_status_level):
        now = time.time()
        if frame is None:
            if aes_key is not None:
                text = "Key synced.\nWaiting for fresh UAV video..."
                hint = "The key is ready. Waiting for the UAV to send a new decodable frame."
                level = "warn"
            else:
                text = "Waiting for UAV key exchange..."
                hint = video_status
                level = video_status_level or "idle"
            self.video_label.config(image="", text=text)
            self.video_hint.config(text=hint, style=self._video_hint_style(level))
            return

        if frame_time is not None and now - frame_time > 1.0:
            self.video_label.config(image="", text="Video stalled.\nWaiting for UAV stream recovery...")
            self.video_hint.config(text=video_status, style=self._video_hint_style("bad"))
            return

        disp = frame.copy()
        if latency is not None:
            cv2.putText(disp, f"Latency={latency:.1f} ms", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w = disp.shape[:2]
        max_w, max_h = 860, 420
        scale = min(max_w / max(w, 1), max_h / max(h, 1))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        disp = cv2.resize(disp, new_size)
        img = Image.fromarray(disp)
        self.video_photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=self.video_photo, text="")
        self.video_hint.config(text=video_status, style=self._video_hint_style(video_status_level))

    @staticmethod
    def _video_hint_style(level):
        if level == "good":
            return "StatusGood.TLabel"
        if level == "warn":
            return "StatusWarn.TLabel"
        if level == "bad":
            return "StatusBad.TLabel"
        return "StatusIdle.TLabel"

    def _update_kdr_chart(self, raw_hist, corr_hist):
        self.kdr_ax.clear()
        self.kdr_ax.set_facecolor("#020617")
        self.kdr_ax.tick_params(colors="#cbd5e1")
        for spine in self.kdr_ax.spines.values():
            spine.set_color("#475569")
        if raw_hist:
            x = list(range(len(raw_hist)))
            self.kdr_ax.plot(x, raw_hist, label="Corrected bits")
        if corr_hist:
            x2 = list(range(len(corr_hist)))
            self.kdr_ax.plot(x2, corr_hist, label="Post-check mismatch")
        self.kdr_ax.set_ylim(0, 100)
        self.kdr_ax.set_title("Correction (%)", color="#e5e7eb")
        self.kdr_ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb") if (raw_hist or corr_hist) else None
        self.kdr_canvas.draw_idle()

    def _update_latency_chart(self, lat_hist, lat_ema_hist):
        self.lat_ax.clear()
        self.lat_ax.set_facecolor("#020617")
        self.lat_ax.tick_params(colors="#cbd5e1")
        for spine in self.lat_ax.spines.values():
            spine.set_color("#475569")
        if lat_hist:
            x = list(range(len(lat_hist)))
            self.lat_ax.plot(x, lat_hist, label="Latency raw", alpha=0.35)
        if lat_ema_hist:
            x_ema = list(range(len(lat_ema_hist)))
            self.lat_ax.plot(x_ema, lat_ema_hist, label="Latency smoothed", linewidth=2.0)
        # if rssi_hist:
        #     x2 = list(range(len(rssi_hist)))
        #     self.lat_ax.plot(x2, rssi_hist, label="RSSI")
        self.lat_ax.set_title("Latency", color="#e5e7eb")
        # if lat_hist or rssi_hist:
        #     self.lat_ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb")
        self.lat_canvas.draw_idle()

    def on_close(self):
        for aid in self.after_ids:
            try:
                self.after_cancel(aid)
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = GSNDashboard()
    app.mainloop()
