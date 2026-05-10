import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse

import random
import threading
import time
import socket
import queue
from collections import deque
from dataclasses import dataclass, field

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import savgol_filter

import sha256
from csi_control import send_csi_reset_request
from key_confirm import verify_key_confirm
from bch_reconciliation import bch_decode_key
from gsn_key_generate import CSISerialWatcher, generate_key
from gsn_receiver import GSNReceiver
try:
    from demo_telemetry import DEMO_TELEMETRY_PORT, parse_telemetry_packet
    DEMO_TELEMETRY_IMPORT_ERROR = None
except ImportError as exc:
    DEMO_TELEMETRY_PORT = 5009
    parse_telemetry_packet = None
    DEMO_TELEMETRY_IMPORT_ERROR = exc

UAV_CONTROL_PORT = 5008
GSN_CSI_PORT = "/dev/cu.usbserial-0001"
EVE_CSI_PORT = "/dev/cu.usbserial-3"
CSI_BAUD = 115200
RAW_HISTORY_LIMIT = 512
EVE_KEY_UPDATE_INTERVAL_SEC = 7.5
EVE_KEY_UPDATE_JITTER_SEC = 2.5
DEMO_TELEMETRY_ENABLED = parse_telemetry_packet is not None
VIDEO_VIEWPORT_MAX_W = 1920
VIDEO_VIEWPORT_MAX_H = 1440
VIDEO_VIEWPORT_MIN_W = 480
VIDEO_VIEWPORT_MIN_H = 360
VIDEO_VIEWPORT_ASPECT = 16 / 9

APP_BG = "#09111f"
SURFACE_BG = "#0f172a"
CARD_BG = "#111c30"
CARD_BG_ALT = "#0c1526"
CARD_BORDER = "#24324a"
CARD_BORDER_SOFT = "#1b2940"
TITLE_BG = "#0a1222"
TEXT_MAIN = "#eef4ff"
TEXT_MUTED = "#8fa2c2"
TEXT_SOFT = "#6f83a6"
ACCENT_BLUE = "#57b4ff"
ACCENT_TEAL = "#3dd7c4"
ACCENT_AMBER = "#ffbf5a"
ACCENT_PINK = "#f472b6"
ACCENT_GREEN = "#4ade80"
ACCENT_VIOLET = "#9b8cff"

GUI_FONT_SCALE_MIN = 0.5
GUI_FONT_SCALE_MAX = 2.0


def _clamp_gui_font_scale(value):
    return max(GUI_FONT_SCALE_MIN, min(GUI_FONT_SCALE_MAX, float(value)))


def _read_gui_font_scale():
    try:
        return _clamp_gui_font_scale(os.environ.get("GSN_GUI_FONT_SCALE", "1.35"))
    except (TypeError, ValueError):
        return 1.0


GUI_FONT_SCALE = _read_gui_font_scale()


def ui_font_size(size):
    return max(6, int(round(size * GUI_FONT_SCALE)))


def ui_font(family, size, *styles):
    return (family, ui_font_size(size), *styles)


def ui_px(size):
    return max(1, int(round(size * GUI_FONT_SCALE)))


def ui_cv_font_scale(scale):
    return max(0.1, float(scale) * GUI_FONT_SCALE)


def ui_cv_thickness(thickness):
    return max(1, int(round(thickness * GUI_FONT_SCALE)))


def ui_cv_fit_font_scale(text, scale, max_width, thickness=1):
    font_scale = ui_cv_font_scale(scale)
    max_width = int(max_width)
    if max_width <= 0:
        return font_scale
    thickness = max(1, int(round(thickness)))
    text_width = cv2.getTextSize(
        str(text),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )[0][0]
    if text_width > max_width:
        font_scale *= max_width / max(text_width, 1)
    return max(0.1, font_scale)


def set_gui_font_scale(value):
    global GUI_FONT_SCALE
    old_scale = GUI_FONT_SCALE
    GUI_FONT_SCALE = _clamp_gui_font_scale(value)
    return old_scale, GUI_FONT_SCALE


def next_eve_key_update_delay():
    jitter = random.uniform(-EVE_KEY_UPDATE_JITTER_SEC, EVE_KEY_UPDATE_JITTER_SEC)
    return max(1.0, EVE_KEY_UPDATE_INTERVAL_SEC + jitter)


def short_bits(bits, limit=64):
    if not bits:
        return "--"
    bits = str(bits)
    return bits if len(bits) <= limit else bits[:limit] + "..."


def fmt_pct(value):
    return "--" if value is None else f"{value:.2f}%"


def wave_summary(values, limit=6):
    if values is None:
        return "--"
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return "--"
    head = ", ".join(f"{v:.3f}" for v in arr[:limit])
    suffix = ", ..." if arr.size > limit else ""
    return f"len={arr.size} [{head}{suffix}]"


def parse_serial_pair(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty serial pair")
    if "," in text:
        return tuple(int(item) for item in text.split(",") if item != "")
    if "-" in text:
        return tuple(int(item) for item in text.split("-") if item != "")
    serial = int(text)
    return (serial,)


def serial_pair_token(pair):
    return ",".join(str(int(item)) for item in parse_serial_pair(pair))


def serial_pair_label(pair):
    parsed = parse_serial_pair(pair)
    return ",".join(str(item) for item in parsed)


@dataclass
class GSNState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    model_loaded: bool = False  # Kept as a UI readiness flag; GSN no longer loads CNN models.
    serial_connected: bool = False
    eve_serial_connected: bool = False
    receiver_started: bool = False
    bch_started: bool = False
    demo_telemetry_started: bool = False

    gsn_raw: str | None = None
    gsn_raw_by_serial: dict = field(default_factory=dict)
    gsn_csi_by_serial: dict = field(default_factory=dict)
    latest_serial: int | None = None
    latest_rssi: float | None = None
    latest_noise: float | None = None
    latest_csi_time: float | None = None
    latest_gsn_live_serial: int | None = None
    latest_gsn_live_csi: np.ndarray | None = None
    latest_gsn_live_csi_time: float | None = None
    latest_eve_serial: int | None = None
    latest_eve_rssi: float | None = None
    latest_eve_noise: float | None = None
    latest_eve_csi: np.ndarray | None = None
    latest_eve_mac: str | None = None
    latest_eve_csi_time: float | None = None
    latest_eve_raw: str | None = None
    active_eve_raw: str | None = None
    active_eve_aes_key: bytes | None = None
    active_eve_serial: int | None = None
    active_eve_key_time: float | None = None
    active_eve_next_update_time: float | None = None
    latest_eve_key_status: str = "Waiting for independent EVE key schedule."
    eve_raw_by_serial: dict = field(default_factory=dict)
    eve_aes_by_serial: dict = field(default_factory=dict)
    pending_uav_csi_reset: bool = True

    last_epoch: int | None = None
    active_key: bytes | None = None
    keys_by_epoch: dict = field(default_factory=dict)
    key_meta_by_epoch: dict = field(default_factory=dict)
    gsn_corrected_by_epoch: dict = field(default_factory=dict)
    epoch_serial_by_epoch: dict = field(default_factory=dict)
    uav_demo_by_epoch: dict = field(default_factory=dict)
    latest_uav_demo: dict | None = None
    latest_demo: dict | None = None
    latest_demo_telemetry_time: float | None = None
    latest_uav_live_serial: int | None = None
    latest_uav_live_csi: np.ndarray | None = None
    latest_uav_live_cnn_csi: np.ndarray | None = None
    latest_uav_live_cnn_serial_pair: tuple | None = None
    latest_uav_live_epoch: int | None = None
    latest_uav_live_csi_time: float | None = None

    latest_latency_ms: float | None = None
    latest_latency_ema_ms: float | None = None
    latest_frame_bgr: np.ndarray | None = None
    latest_frame_time: float | None = None
    latest_eve_video_bgr: np.ndarray | None = None
    latest_eve_video_time: float | None = None
    latest_eve_video_encrypted: bool | None = None
    latest_eve_video_decrypted: bool | None = None
    video_encryption_enabled: bool = True
    latest_uav_ip: str | None = None
    video_status: str = "Waiting for frames..."
    video_status_level: str = "idle"

    latest_kdr_raw: float | None = None
    latest_kdr_corr: float | None = None
    latest_demo_raw_kdr: float | None = None
    latest_demo_cnn_kdr: float | None = None
    latest_demo_cnnq_kdr: float | None = None
    kdr_raw_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    kdr_corr_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    demo_raw_kdr_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    demo_cnn_kdr_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    demo_cnnq_kdr_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    demo_hist_epochs: set = field(default_factory=set)
    latency_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    latency_ema_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    rssi_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    uav_rssi_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    noise_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    eve_rssi_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    eve_noise_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    hist_idx: int = 0


class PanelHost(tk.Frame):
    def __init__(self, parent, name, placeholder):
        super().__init__(parent, bg=SURFACE_BG, highlightthickness=0, bd=0, padx=8, pady=8)
        self.name = name
        self.placeholder_text = placeholder
        self.panels = []
        self.stack = tk.Frame(self, bg=SURFACE_BG, highlightthickness=0, bd=0)
        self.stack.pack(fill="both", expand=True)
        self.placeholder = tk.Label(
            self.stack,
            text=self.placeholder_text,
            bg=CARD_BG_ALT,
            fg=TEXT_MUTED,
            font=ui_font("Arial", 10, "italic"),
            highlightthickness=1,
            highlightbackground=CARD_BORDER_SOFT,
        )
        self.placeholder.pack(fill="both", expand=True, padx=8, pady=8)
        self._tag_host_widgets()

    def _tag_host_widgets(self):
        self._panel_host_ref = self
        self.stack._panel_host_ref = self
        self.placeholder._panel_host_ref = self

    def add_panel(self, panel):
        if panel not in self.panels:
            self.panels.append(panel)
        self._refresh_placeholder()

    def remove_panel(self, panel):
        if panel in self.panels:
            self.panels.remove(panel)
        self._refresh_placeholder()

    def _refresh_placeholder(self):
        if self.panels:
            if self.placeholder.winfo_exists():
                self.placeholder.pack_forget()
        else:
            self.placeholder.pack(fill="both", expand=True, padx=8, pady=8)

    def set_drop_highlight(self, active):
        self.configure(highlightthickness=1 if active else 0, highlightbackground=ACCENT_BLUE)


class BasePanel:
    def __init__(self, dashboard, key, title):
        self.dashboard = dashboard
        self.key = key
        self.title = title
        self.host = None
        self.last_host_name = None
        self.visible = True
        self.card = None
        self.body = None

    def mount(self, host):
        if self.host is host and self.card is not None:
            return
        self.unmount()
        self.host = host
        self.last_host_name = host.name
        host.add_panel(self)

        self.card = tk.Frame(host.stack, bg=CARD_BG, highlightthickness=1, highlightbackground=CARD_BORDER, bd=0)
        self.card.pack(fill="both", expand=True, padx=4, pady=4)
        titlebar = tk.Frame(self.card, bg=TITLE_BG, height=30)
        titlebar.pack(fill="x")
        titlebar.pack_propagate(False)
        grip = tk.Label(titlebar, text="::", bg=TITLE_BG, fg=TEXT_SOFT, font=ui_font("Consolas", 11, "bold"))
        grip.pack(side="left", padx=(8, 4))
        label = tk.Label(titlebar, text=self.title, bg=TITLE_BG, fg=TEXT_MAIN, font=ui_font("Arial", 11, "bold"))
        label.pack(side="left", padx=(0, 8))
        self.body = tk.Frame(self.card, bg=CARD_BG)
        self.body.pack(fill="both", expand=True)

        for widget in (titlebar, grip, label):
            self.dashboard.bind_panel_drag(self, widget)

        self.build_body(self.body)
        self.render()

    def unmount(self):
        if self.host is not None:
            self.host.remove_panel(self)
        if self.card is not None:
            self.card.destroy()
        self.host = None
        self.card = None
        self.body = None

    def show(self):
        self.visible = True
        target = self.dashboard.panel_hosts.get(self.last_host_name) or self.dashboard.default_hosts.get(self.key)
        if target is not None:
            self.mount(target)

    def hide(self):
        self.visible = False
        self.unmount()

    def build_body(self, parent):
        raise NotImplementedError

    def render(self):
        pass


class ModulePanel(BasePanel):
    def __init__(self, dashboard, key, title, content_options, default_content):
        super().__init__(dashboard, key, title)
        self.content_options = dict(content_options)
        self.default_content = default_content
        self.content_key = default_content
        self.snapshot = {}

    def mount(self, host):
        if self.host is host and self.card is not None:
            return
        self.unmount()
        self.host = host
        self.last_host_name = host.name
        host.add_panel(self)

        self.card = tk.Frame(host.stack, bg=CARD_BG, highlightthickness=1, highlightbackground=CARD_BORDER, bd=0)
        self.card.pack(fill="both", expand=True, padx=6, pady=6)
        titlebar = tk.Frame(self.card, bg=TITLE_BG, height=34)
        titlebar.pack(fill="x")
        titlebar.pack_propagate(False)
        accent = tk.Frame(titlebar, bg=ACCENT_BLUE, width=3)
        accent.pack(side="left", fill="y", padx=(0, 8))
        grip = tk.Label(titlebar, text="::", bg=TITLE_BG, fg=TEXT_SOFT, font=ui_font("Consolas", 12, "bold"))
        grip.pack(side="left", padx=(0, 7))
        label = tk.Label(titlebar, text=self.title, bg=TITLE_BG, fg=TEXT_MAIN, font=ui_font("Arial", 11, "bold"))
        label.pack(side="left", fill="x", expand=True, padx=(0, 8))
        if len(self.content_options) > 1:
            self.content_var = tk.StringVar(value=self.content_options[self.content_key])
            selector = ttk.Combobox(
                titlebar,
                state="readonly",
                width=18,
                values=list(self.content_options.values()),
                textvariable=self.content_var,
                style="Panel.TCombobox",
            )
            selector.pack(side="right", padx=10, pady=5)
            selector.bind("<<ComboboxSelected>>", self._on_content_change)
            self.selector = selector
        else:
            self.content_var = None
            self.selector = None
            tk.Label(titlebar, text=self.content_options[self.content_key], bg=TITLE_BG, fg=TEXT_MUTED, font=ui_font("Arial", 9, "bold")).pack(side="right", padx=10)

        self.body = tk.Frame(self.card, bg=CARD_BG)
        self.body.pack(fill="both", expand=True)
        tk.Frame(self.card, bg=CARD_BORDER_SOFT, height=1).pack(fill="x")

        for widget in (titlebar, grip, label):
            self.dashboard.bind_panel_drag(self, widget)

        self.build_body(self.body)
        self.render()

    def _on_content_change(self, _event=None):
        selected_label = self.content_var.get()
        for key, label in self.content_options.items():
            if label == selected_label:
                self.content_key = key
                break
        self.render()

    def update_snapshot(self, snapshot):
        self.snapshot = snapshot
        self.render()


class VideoModulePanel(ModulePanel):
    def __init__(self, dashboard, key="media_main", title="Video Stream", default_content="video"):
        super().__init__(
            dashboard,
            key,
            title,
            {"video": "UAV / EVE View"},
            default_content,
        )
        self.video_photo = None
        self.viewport_size = (VIDEO_VIEWPORT_MAX_W, VIDEO_VIEWPORT_MAX_H)

    def build_body(self, parent):
        self.viewport_outer = tk.Frame(parent, bg="#111827", highlightthickness=0)
        self.viewport_outer.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        self.viewport = tk.Frame(
            self.viewport_outer,
            bg="#020617",
            width=VIDEO_VIEWPORT_MAX_W,
            height=VIDEO_VIEWPORT_MAX_H,
            highlightthickness=0,
        )
        self.viewport.place(relx=0.5, rely=0.5, anchor="center")
        self.viewport.pack_propagate(False)
        self.video_label = tk.Label(
            self.viewport,
            text="Waiting for frames...",
            bg="#020617",
            fg="#cbd5e1",
            font=ui_font("Arial", 12, "bold"),
        )
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_hint = ttk.Label(parent, text="Waiting for UAV video stream.", style="StatusIdle.TLabel")
        self.video_hint.pack(anchor="w", padx=10, pady=(8, 10))
        self.viewport_outer.bind("<Configure>", self._on_configure)
        self.dashboard.after(0, self._sync_viewport)

    def _on_configure(self, _event):
        self._sync_viewport()

    def _sync_viewport(self):
        if self.card is None:
            return
        outer_w = self.viewport_outer.winfo_width()
        outer_h = self.viewport_outer.winfo_height()
        if outer_w <= 1 or outer_h <= 1:
            return

        target_w = min(outer_w, VIDEO_VIEWPORT_MAX_W)
        target_h = int(target_w / VIDEO_VIEWPORT_ASPECT)
        if target_h > outer_h:
            target_h = min(outer_h, VIDEO_VIEWPORT_MAX_H)
            target_w = int(target_h * VIDEO_VIEWPORT_ASPECT)

        target_w = max(VIDEO_VIEWPORT_MIN_W, target_w)
        target_h = max(VIDEO_VIEWPORT_MIN_H, target_h)
        if target_w > outer_w or target_h > outer_h:
            scale = min(outer_w / max(target_w, 1), outer_h / max(target_h, 1))
            target_w = max(1, int(target_w * scale))
            target_h = max(1, int(target_h * scale))

        self.viewport_size = (target_w, target_h)
        self.viewport.place_configure(width=target_w, height=target_h)
        self.render()

    @staticmethod
    def _placeholder_pip(width, height, text):
        pip = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
        pip[:] = (24, 31, 45)
        thickness = ui_cv_thickness(1)
        cv2.putText(
            pip,
            text,
            (ui_px(10), max(ui_px(28), height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            ui_cv_fit_font_scale(text, 0.55, width - ui_px(20), thickness),
            (203, 213, 225),
            thickness,
            cv2.LINE_AA,
        )
        return pip

    def _compose_with_eve_pip(self, frame):
        disp = frame.copy()
        h, w = disp.shape[:2]
        if h <= 0 or w <= 0:
            return disp

        eve_frame = self.snapshot.get("eve_video_frame")
        encrypted = self.snapshot.get("eve_video_encrypted")
        eve_decrypted = self.snapshot.get("eve_video_decrypted")
        encryption_enabled = bool(self.snapshot.get("video_encryption_enabled", True))
        eve_time = self.snapshot.get("eve_video_time")
        eve_is_fresh = eve_time is not None and time.time() - eve_time <= 1.5

        pip_w = int(w * 0.5)
        pip_w = max(250, min(pip_w, 340, int(w * 0.5)))
        pip_h = int(pip_w / VIDEO_VIEWPORT_ASPECT)
        pip_h = max(140, min(pip_h, int(h * 0.5)))
        pip_w = max(1, min(pip_w, w - 24))
        pip_h = max(1, min(pip_h, h - 24))

        if eve_frame is None or not eve_is_fresh:
            pip = self._placeholder_pip(pip_w, pip_h, "EVE waiting")
            label = "EVE"
            color = (203, 213, 225)
        else:
            pip = cv2.resize(eve_frame, (pip_w, pip_h))
            if encrypted:
                if eve_decrypted:
                    label = "EVE (decrypted)"
                    color = (80, 255, 180)
                else:
                    label = "EVE (could not decrypt)"
                    color = (80, 180, 255)
            else:
                label = "EVE (no encryption)"
                color = (80, 255, 180)

        margin = max(ui_px(12), int(min(w, h) * 0.025))
        x1 = max(0, w - pip_w - margin)
        y1 = max(0, h - pip_h - margin)
        x2 = min(w, x1 + pip_w)
        y2 = min(h, y1 + pip_h)
        pip = pip[: y2 - y1, : x2 - x1]

        border = ui_px(3)
        cv2.rectangle(
            disp,
            (max(0, x1 - border), max(0, y1 - border)),
            (min(w - 1, x2 + border), min(h - 1, y2 + border)),
            color,
            border,
        )
        disp[y1:y2, x1:x2] = pip

        label_h = ui_px(26)
        overlay = disp.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, min(y2, y1 + label_h)), (2, 6, 23), -1)
        cv2.addWeighted(overlay, 0.68, disp, 0.32, 0, disp)
        label_thickness = ui_cv_thickness(1)
        cv2.putText(
            disp,
            label,
            (x1 + ui_px(8), y1 + ui_px(19)),
            cv2.FONT_HERSHEY_SIMPLEX,
            ui_cv_fit_font_scale(label, 0.55, (x2 - x1) - ui_px(16), label_thickness),
            color,
            label_thickness,
            cv2.LINE_AA,
        )

        mode = "Encrypted" if encryption_enabled else "Plaintext"
        # cv2.putText(
        #     disp,
        #     mode,
        #     (10, max(28, h - 16)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.65,
        #     (226, 232, 240),
        #     2,
        #     cv2.LINE_AA,
        # )
        return disp

    def _present_bgr_frame(self, disp, hint, level):
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w = disp.shape[:2]
        max_w, max_h = self.viewport_size
        scale = min(max_w / max(w, 1), max_h / max(h, 1))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        disp = cv2.resize(disp, new_size)
        img = Image.fromarray(disp)
        self.video_photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=self.video_photo, text="")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_hint.config(text=hint, style=self.dashboard.video_hint_style(level))

    @staticmethod
    def _draw_stalled_overlay(disp):
        h, w = disp.shape[:2]
        overlay = disp.copy()
        edge_pad = ui_px(12)
        box_w = min(max(ui_px(360), int(w * 0.46)), max(1, w - ui_px(32)))
        box_h = ui_px(82)
        x1 = max(edge_pad, (w - box_w) // 2)
        y1 = max(edge_pad, int(h * 0.12))
        x2 = min(w - edge_pad, x1 + box_w)
        y2 = min(h - edge_pad, y1 + box_h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (2, 6, 23), -1)
        cv2.addWeighted(overlay, 0.72, disp, 0.28, 0, disp)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (251, 113, 133), ui_cv_thickness(2))
        title = "Video stalled"
        title_thickness = ui_cv_thickness(2)
        cv2.putText(
            disp,
            title,
            (x1 + ui_px(18), y1 + ui_px(32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            ui_cv_fit_font_scale(title, 0.82, (x2 - x1) - ui_px(36), title_thickness),
            (251, 113, 133),
            title_thickness,
            cv2.LINE_AA,
        )
        subtitle = "Waiting for UAV stream recovery"
        subtitle_thickness = ui_cv_thickness(1)
        cv2.putText(
            disp,
            subtitle,
            (x1 + ui_px(18), y1 + ui_px(62)),
            cv2.FONT_HERSHEY_SIMPLEX,
            ui_cv_fit_font_scale(subtitle, 0.55, (x2 - x1) - ui_px(36), subtitle_thickness),
            (226, 232, 240),
            subtitle_thickness,
            cv2.LINE_AA,
        )
        return disp

    def render(self):
        if self.card is None:
            return

        frame = self.snapshot.get("frame")
        latency = self.snapshot.get("latency_ema")
        frame_time = self.snapshot.get("frame_time")
        aes_key = self.snapshot.get("aes_key")
        video_status = self.snapshot.get("video_status") or "Waiting for UAV video stream."
        video_status_level = self.snapshot.get("video_status_level") or "idle"
        now = time.time()

        if frame is None:
            if aes_key is not None:
                text = "Key synced.\nWaiting for fresh UAV video..."
                hint = "The key is ready. Waiting for the UAV to send a new decodable frame."
                level = "warn"
            else:
                text = "Waiting for UAV key exchange..."
                hint = "Need a confirmed epoch before video can be decrypted."
                level = "idle"
            if video_status:
                hint = video_status
                level = video_status_level
            self.video_label.config(image="", text=text)
            self.video_label.place(relx=0.5, rely=0.5, anchor="center")
            self.video_hint.config(text=hint, style=self.dashboard.video_hint_style(level))
            return

        if frame_time is not None and now - frame_time > 1.0:
            disp = self._compose_with_eve_pip(frame)
            disp = self._draw_stalled_overlay(disp)
            self._present_bgr_frame(disp, video_status, "bad")
            return

        disp = self._compose_with_eve_pip(frame)
        if latency is not None:
            latency_text = f"Latency={latency:.1f} ms"
            latency_thickness = ui_cv_thickness(2)
            cv2.putText(
                disp,
                latency_text,
                (ui_px(10), ui_px(28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                ui_cv_fit_font_scale(latency_text, 0.8, disp.shape[1] - ui_px(20), latency_thickness),
                (0, 255, 0),
                latency_thickness,
                cv2.LINE_AA,
            )
        self._present_bgr_frame(disp, video_status, video_status_level)


class TextModulePanel(ModulePanel):
    TEXT_OPTIONS = {
        "key_status": "Key Status",
        "demo_keys": "Demo Keys",
        "epoch_history": "Epoch History",
        "log": "System Log",
        "link_snapshot": "Link Snapshot",
    }

    def __init__(self, dashboard, key, title, default_content):
        super().__init__(dashboard, key, title, self.TEXT_OPTIONS, default_content)

    def build_body(self, parent):
        self.text = tk.Text(
            parent,
            wrap="word",
            bg="#020617",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            font=ui_font("Consolas", 10),
        )
        self.text.pack(fill="both", expand=True, padx=10, pady=10)
        self.text.config(state="disabled")

    @staticmethod
    def _eve_video_summary(encrypted, decrypted):
        if encrypted is False:
            return "plaintext stream"
        if encrypted is True:
            return "decrypted with EVE key" if decrypted else "noise / key mismatch"
        return "--"

    def render(self):
        if self.card is None:
            return

        serial = self.snapshot.get("serial")
        rssi = self.snapshot.get("rssi")
        noise = self.snapshot.get("noise")
        gsn_raw = self.snapshot.get("gsn_raw")
        gsn_live_serial = self.snapshot.get("gsn_live_serial")
        gsn_live_csi = self.snapshot.get("gsn_live_csi")
        uav_live_serial = self.snapshot.get("uav_live_serial")
        uav_live_csi = self.snapshot.get("uav_live_csi")
        eve_serial = self.snapshot.get("eve_serial")
        eve_rssi = self.snapshot.get("eve_rssi")
        eve_noise = self.snapshot.get("eve_noise")
        eve_mac = self.snapshot.get("eve_mac")
        eve_csi = self.snapshot.get("eve_csi")
        eve_raw = self.snapshot.get("eve_raw")
        eve_key_status = self.snapshot.get("eve_key_status")
        eve_video_encrypted = self.snapshot.get("eve_video_encrypted")
        eve_video_decrypted = self.snapshot.get("eve_video_decrypted")
        epoch = self.snapshot.get("epoch")
        aes_key = self.snapshot.get("aes_key")
        video_encryption_enabled = self.snapshot.get("video_encryption_enabled")
        latest_uav_ip = self.snapshot.get("latest_uav_ip")
        keys_by_epoch = self.snapshot.get("keys_by_epoch", {})
        latency = self.snapshot.get("latency")
        latency_ema = self.snapshot.get("latency_ema")
        raw_kdr = self.snapshot.get("raw_kdr")
        corr_kdr = self.snapshot.get("corr_kdr")
        demo = self.snapshot.get("demo")
        uav_demo = self.snapshot.get("uav_demo")
        demo_raw_kdr = self.snapshot.get("demo_raw_kdr")
        demo_cnn_kdr = self.snapshot.get("demo_cnn_kdr")
        demo_cnnq_kdr = self.snapshot.get("demo_cnnq_kdr")
        video_status = self.snapshot.get("video_status")

        if self.content_key == "key_status":
            lines = [
                f"serial       : {serial if serial is not None else '--'}",
                f"UAV live seq : {uav_live_serial if uav_live_serial is not None else '--'}",
                f"GSN live seq : {gsn_live_serial if gsn_live_serial is not None else '--'}",
                f"rssi/noise   : {('--' if rssi is None else f'{rssi:.1f}')}/{('--' if noise is None else f'{noise:.1f}')}",
                f"raw key      : {short_bits(gsn_raw)}",
                f"video mode   : {'encrypted' if video_encryption_enabled else 'plaintext'}",
                f"EVE seq      : {eve_serial if eve_serial is not None else '--'}",
                f"EVE rssi/noise: {('--' if eve_rssi is None else f'{eve_rssi:.1f}')}/{('--' if eve_noise is None else f'{eve_noise:.1f}')}",
                f"EVE raw key  : {short_bits(eve_raw)}",
                f"EVE key      : {eve_key_status or '--'}",
                f"EVE video    : {self._eve_video_summary(eve_video_encrypted, eve_video_decrypted)}",
                f"active epoch : {epoch if epoch is not None else '--'}",
                f"aes key      : {(aes_key.hex()[:48] + '...') if isinstance(aes_key, (bytes, bytearray)) else '--'}",
                f"raw KDR      : {fmt_pct(demo_raw_kdr)}",
                f"CNN KDR      : {fmt_pct(demo_cnn_kdr)}",
                f"CNN-Q KDR    : {fmt_pct(demo_cnnq_kdr)}",
            ]
        elif self.content_key == "demo_keys":
            if not demo and not uav_demo:
                lines = ["Waiting for demo telemetry..."]
            elif not demo:
                pair_label = serial_pair_label(uav_demo.get("serial_pair", (uav_demo.get("serial", "--"),)))
                lines = [
                    "UAV demo telemetry received.",
                    "Waiting for matching GSN serial and confirmed BCH epoch...",
                    "",
                    f"epoch/serials    : {uav_demo.get('epoch', '--')}/{pair_label}",
                    f"UAV RSSI         : {uav_demo.get('uav_rssi', ['--'])[0]}",
                    f"UAV raw key      : {short_bits(uav_demo.get('uav_raw_key'))}",
                    f"UAV raw key 2    : {short_bits(uav_demo.get('uav_raw_key_2'))}",
                    f"UAV CNN key      : {short_bits(uav_demo.get('uav_cnn_key'))}",
                    f"UAV CNN-Q key    : {short_bits(uav_demo.get('uav_cnnq_key'))}",
                    f"UAV active key   : {short_bits(uav_demo.get('uav_corrected_key'))}",
                    f"UAV raw CSI      : {wave_summary(uav_demo.get('uav_raw_csi'))}",
                    f"UAV raw CSI 2    : {wave_summary(uav_demo.get('uav_raw_csi_2'))}",
                    f"UAV CNN CSI      : {wave_summary(uav_demo.get('uav_cnn_csi'))}",
                    "",
                    f"raw KDR          : {fmt_pct(demo_raw_kdr)}",
                    f"CNN KDR          : {fmt_pct(demo_cnn_kdr)}",
                    f"CNN-Q KDR        : {fmt_pct(demo_cnnq_kdr)}",
                ]
            else:
                pair_label = serial_pair_label(demo.get("serial_pair", (demo.get("serial", "--"),)))
                lines = [
                    f"epoch/serials    : {demo.get('epoch', '--')}/{pair_label}",
                    f"UAV RSSI         : {demo.get('uav_rssi', ['--'])[0]}",
                    f"raw KDR          : {fmt_pct(demo.get('raw_kdr'))}",
                    f"CNN KDR          : {fmt_pct(demo.get('cnn_kdr'))}",
                    f"CNN-Q KDR        : {fmt_pct(demo.get('cnnq_kdr'))}",
                    "",
                    f"UAV raw key      : {short_bits(demo.get('uav_raw_key'))}",
                    f"UAV raw key 2    : {short_bits(demo.get('uav_raw_key_2'))}",
                    f"GSN raw key      : {short_bits(demo.get('gsn_raw_key'))}",
                    f"GSN raw key 2    : {short_bits(demo.get('gsn_raw_key_2'))}",
                    f"UAV CNN key      : {short_bits(demo.get('uav_cnn_key'))}",
                    f"UAV CNN-Q key    : {short_bits(demo.get('uav_cnnq_key'))}",
                    f"UAV active key   : {short_bits(demo.get('uav_corrected_key'))}",
                    f"GSN corrected key: {short_bits(demo.get('gsn_corrected_key'))}",
                    "",
                    f"UAV raw CSI      : {wave_summary(demo.get('uav_raw_csi'))}",
                    f"UAV raw CSI 2    : {wave_summary(demo.get('uav_raw_csi_2'))}",
                    f"UAV CNN CSI      : {wave_summary(demo.get('uav_cnn_csi'))}",
                    f"GSN raw CSI      : {wave_summary(demo.get('gsn_raw_csi'))}",
                    f"GSN raw CSI 2    : {wave_summary(demo.get('gsn_raw_csi_2'))}",
                ]
        elif self.content_key == "epoch_history":
            lines = []
            for item_epoch in sorted(keys_by_epoch.keys(), reverse=True)[:20]:
                aes = keys_by_epoch[item_epoch]
                key_str = aes.hex()[:32] + "..." if isinstance(aes, (bytes, bytearray)) else str(aes)
                lines.append(f"epoch {item_epoch:<6} key {key_str}")
            if not lines:
                lines = ["No epochs yet."]
        elif self.content_key == "link_snapshot":
            lines = [
                f"serial        : {serial if serial is not None else '--'}",
                f"active epoch  : {epoch if epoch is not None else '--'}",
                f"rssi          : {('--' if rssi is None else f'{rssi:.1f}')}",
                f"noise         : {('--' if noise is None else f'{noise:.1f}')}",
                f"UAV live seq  : {uav_live_serial if uav_live_serial is not None else '--'}",
                f"UAV live CSI  : {wave_summary(uav_live_csi)}",
                f"GSN live seq  : {gsn_live_serial if gsn_live_serial is not None else '--'}",
                f"GSN live CSI  : {wave_summary(gsn_live_csi)}",
                f"EVE seq       : {eve_serial if eve_serial is not None else '--'}",
                f"EVE mac       : {eve_mac or '--'}",
                f"EVE rssi      : {('--' if eve_rssi is None else f'{eve_rssi:.1f}')}",
                f"EVE noise     : {('--' if eve_noise is None else f'{eve_noise:.1f}')}",
                f"EVE CSI       : {wave_summary(eve_csi)}",
                f"EVE raw key   : {short_bits(eve_raw)}",
                f"EVE key       : {eve_key_status or '--'}",
                f"EVE video     : {self._eve_video_summary(eve_video_encrypted, eve_video_decrypted)}",
                f"latency raw   : {('--' if latency is None else f'{latency:.1f} ms')}",
                f"latency avg   : {('--' if latency_ema is None else f'{latency_ema:.1f} ms')}",
                f"raw KDR       : {fmt_pct(demo_raw_kdr)}",
                f"CNN KDR       : {fmt_pct(demo_cnn_kdr)}",
                f"CNN-Q KDR     : {fmt_pct(demo_cnnq_kdr)}",
                f"corr bits     : {('--' if raw_kdr is None else f'{raw_kdr:.2f}%')}",
                f"post-check    : {('--' if corr_kdr is None else f'{corr_kdr:.2f}%')}",
                f"video status  : {video_status or '--'}",
                f"UAV control IP: {latest_uav_ip or 'broadcast'}",
            ]
        else:
            lines = list(self.dashboard.log_lines)
            if not lines:
                lines = ["System log is empty."]

        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", "\n".join(lines))
        if self.content_key == "log":
            self.text.see("end")
        self.text.config(state="disabled")


class ChartModulePanel(ModulePanel):
    CHART_OPTIONS = {
        "latency": "Latency",
        "correction": "Correction Trend",
        "demo_kdr": "Demo KDR",
        "csi": "CSI Waveform",
        "live_csi": "Live CSI",
        "signal": "RSSI / Noise",
    }

    def __init__(self, dashboard, key, title, default_content):
        super().__init__(dashboard, key, title, self.CHART_OPTIONS, default_content)

    def build_body(self, parent):
        fig = Figure(figsize=(4.2, 1.8), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#020617")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.fig = fig
        self.ax = ax
        self.canvas = canvas

    @staticmethod
    def _normalized_wave(values):
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi - lo < 1e-9:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    def render(self):
        if self.card is None:
            return

        self.ax.clear()
        self.ax.set_facecolor("#020617")
        self.ax.tick_params(colors="#cbd5e1", labelsize=ui_font_size(8))
        for spine in self.ax.spines.values():
            spine.set_color("#475569")

        if self.content_key == "latency":
            self.ax.set_ylim(0, 100)
            self.ax.set_xticks([])
            lat_hist = list(self.snapshot.get("lat_hist", []))
            lat_ema_hist = list(self.snapshot.get("lat_ema_hist", []))
            # if lat_hist:
            #     x = list(range(len(lat_hist)))
            #     self.ax.plot(x, lat_hist, label="Latency raw", alpha=0.35)
            if lat_ema_hist:
                x_ema = list(range(len(lat_ema_hist)))
                self.ax.plot(x_ema, lat_ema_hist, label="Latency", linewidth=2.0)
            self.ax.set_title("Latency", color="#e5e7eb", fontsize=ui_font_size(10))
            if lat_hist or lat_ema_hist:
                self.ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb", fontsize=ui_font_size(8))

        elif self.content_key == "correction":
            raw_hist = list(self.snapshot.get("kdr_raw_hist", []))
            corr_hist = list(self.snapshot.get("kdr_corr_hist", []))
            if raw_hist:
                x = list(range(len(raw_hist)))
                self.ax.plot(x, raw_hist, label="Corrected bits")
            if corr_hist:
                x2 = list(range(len(corr_hist)))
                self.ax.plot(x2, corr_hist, label="Post-check mismatch")
            self.ax.set_ylim(0, 100)
            self.ax.set_title("Correction (%)", color="#e5e7eb", fontsize=ui_font_size(10))
            if raw_hist or corr_hist:
                self.ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb", fontsize=ui_font_size(8))

        elif self.content_key == "demo_kdr":
            raw_hist = list(self.snapshot.get("demo_raw_kdr_hist", []))
            cnn_hist = list(self.snapshot.get("demo_cnn_kdr_hist", []))
            cnnq_hist = list(self.snapshot.get("demo_cnnq_kdr_hist", []))
            if raw_hist:
                x = list(range(len(raw_hist)))
                self.ax.plot(x, raw_hist, label="Raw key KDR")
            if cnn_hist:
                x2 = list(range(len(cnn_hist)))
                self.ax.plot(x2, cnn_hist, label="CNN key KDR")
            if cnnq_hist:
                x3 = list(range(len(cnnq_hist)))
                self.ax.plot(x3, cnnq_hist, label="CNN-Q key KDR")
            self.ax.set_ylim(0, 100)
            self.ax.set_title("Demo Key KDR (%)", color="#e5e7eb", fontsize=ui_font_size(10))
            if raw_hist or cnn_hist or cnnq_hist:
                self.ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb", fontsize=ui_font_size(8))

        elif self.content_key == "csi":
            demo = self.snapshot.get("demo") or {}
            uav_demo = self.snapshot.get("uav_demo") or {}
            series = [
                ("UAV raw CSI", demo.get("uav_raw_csi") or uav_demo.get("uav_raw_csi")),
                # ("UAV raw CSI 2", demo.get("uav_raw_csi_2") or uav_demo.get("uav_raw_csi_2")),
                ("UAV CNN CSI", demo.get("uav_cnn_csi") or uav_demo.get("uav_cnn_csi")),
                ("GSN raw CSI", demo.get("gsn_raw_csi")),
                # ("GSN raw CSI 2", demo.get("gsn_raw_csi_2")),
                ("EVE CSI", demo.get("eve_csi")),
            ]
            plotted = False
            for label, values in series:
                wave = values
                wave = savgol_filter(wave, min(9, len(wave) // 2 * 2 - 1), 3) if wave is not None and len(wave) >= 5 else wave
                wave = self._normalized_wave(wave)
                if wave is None:
                    continue
                self.ax.plot(range(len(wave)), wave, label=label, linewidth=1.7)
                plotted = True
            self.ax.set_ylim(-0.1, 1.3)
            self.ax.set_yticks([])
            self.ax.set_title("CSI Waveform", color="#e5e7eb", fontsize=ui_font_size(10))
            if plotted:
                self.ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb", fontsize=ui_font_size(8), ncol=2, loc="upper right")

        elif self.content_key == "live_csi":
            series = [
                ("UAV live CSI", self.snapshot.get("uav_live_csi")),
                ("UAV CNN CSI", self.snapshot.get("uav_live_cnn_csi")),
                ("GSN live CSI", self.snapshot.get("gsn_live_csi")),
                ("EVE live CSI", self.snapshot.get("eve_csi")),
            ]
            plotted = False
            for label, values in series:
                wave = values
                wave = savgol_filter(wave, min(9, len(wave) // 2 * 2 - 1), 3) if wave is not None and len(wave) >= 5 else wave
                wave = self._normalized_wave(wave)
                if wave is None:
                    continue
                self.ax.plot(range(len(wave)), wave, label=label, linewidth=1.8)
                plotted = True
            self.ax.set_ylim(-0.1, 1.3)
            self.ax.set_title("Live CSI Waveform", color="#e5e7eb", fontsize=ui_font_size(10))
            self.ax.set_yticks([])
            if plotted:
                self.ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb", fontsize=ui_font_size(8), ncol=2, loc="upper right")

        else:
            self.ax.set_xlim(0, 50)
            # self.ax.set_ylim(-100, 0)
            self.ax.set_xticks([])
            rssi_hist = list(self.snapshot.get("rssi_hist", []))
            uav_rssi_hist = list(self.snapshot.get("uav_rssi_hist", []))
            eve_rssi_hist = list(self.snapshot.get("eve_rssi_hist", []))
            # ema
            rssi_ema_hist = []
            uav_rssi_ema_hist = []
            eve_rssi_ema_hist = []
            for rssi in rssi_hist:
                if rssi_ema_hist:
                    new_ema = 0.3 * rssi + 0.7 * rssi_ema_hist[-1]
                else:
                    new_ema = rssi
                rssi_ema_hist.append(new_ema)
            for uav_rssi in uav_rssi_hist:
                if uav_rssi_ema_hist:
                    new_ema = 0.3 * uav_rssi + 0.7 * uav_rssi_ema_hist[-1]
                else:
                    new_ema = uav_rssi
                uav_rssi_ema_hist.append(new_ema)
            for eve_rssi in eve_rssi_hist:
                if eve_rssi_ema_hist:
                    new_ema = 0.3 * eve_rssi + 0.7 * eve_rssi_ema_hist[-1]
                else:
                    new_ema = eve_rssi
                eve_rssi_ema_hist.append(new_ema)
            if rssi_hist:
                x = list(range(len(rssi_hist)))
                self.ax.plot(x, rssi_ema_hist, label="GSN RSSI", linewidth=2.0)
            if uav_rssi_hist:
                x2 = list(range(len(uav_rssi_hist)))
                self.ax.plot(x2, uav_rssi_ema_hist, label="UAV RSSI", linewidth=1.7, alpha=0.9)
            if eve_rssi_hist:
                x3 = list(range(len(eve_rssi_hist)))
                self.ax.plot(x3, eve_rssi_ema_hist, label="EVE RSSI", linewidth=1.7, alpha=0.9)
            self.ax.set_title("RSSI / Noise", color="#e5e7eb", fontsize=ui_font_size(10))
            if rssi_hist or uav_rssi_hist or eve_rssi_hist:
                self.ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb", fontsize=ui_font_size(8))

        self.canvas.draw_idle()


class ControlModulePanel(ModulePanel):
    CONTROL_OPTIONS = {"session": "Runtime State"}

    def __init__(self, dashboard, key="control_panel", title="System State", default_content="session"):
        super().__init__(dashboard, key, title, self.CONTROL_OPTIONS, default_content)

    def build_body(self, parent):
        self.summary = tk.Text(
            parent,
            height=7,
            wrap="word",
            bg="#020617",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            font=ui_font("Consolas", 10),
        )
        self.summary.pack(fill="both", expand=True, padx=10, pady=10)
        self.summary.config(state="disabled")

    def render(self):
        if self.card is None:
            return

        lines = [
            f"backend      : {'ON' if self.snapshot.get('started') else 'OFF'}",
            f"keygen ready : {'ON' if self.snapshot.get('model_loaded') else 'OFF'}",
            f"serial link  : {'ON' if self.snapshot.get('serial_ok') else 'OFF'}",
            f"EVE serial   : {'ON' if self.snapshot.get('eve_serial_ok') else 'OFF'}",
            f"video cipher : {'ON' if self.snapshot.get('video_encryption_enabled') else 'OFF'}",
            f"video rx     : {'ON' if self.snapshot.get('rx_ok') else 'OFF'}",
            f"bch rx       : {'ON' if self.snapshot.get('bch_ok') else 'OFF'}",
            f"demo rx      : {'ON' if self.snapshot.get('demo_ok') else ('OFF' if self.snapshot.get('demo_enabled') else 'DISABLED')}",
            f"demo packet  : {'SEEN' if self.snapshot.get('uav_demo') else '--'}",
            f"ui refresh   : {'PAUSED' if self.dashboard.ui_paused else 'RUNNING'}",
        ]
        self.summary.config(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", "\n".join(lines))
        self.summary.config(state="disabled")


class StatsStrip:
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=SURFACE_BG)
        self.frame.pack(fill="x", pady=(0, 10))
        self.cards = {}
        self.kdr_value_labels = {}
        self.card_keys = []
        self.card_widgets = []
        self.current_columns = None
        specs = [
            ("serial", "Latest Serial", "Stream sequence", ACCENT_BLUE),
            ("rssi", "RSSI", "Radio strength", ACCENT_TEAL),
            ("noise", "Noise", "Channel floor", ACCENT_AMBER),
            # ("epoch", "Active Epoch", "Confirmed key window", ACCENT_VIOLET),
            ("latency", "Latency (ms)", "End-to-end response", ACCENT_PINK),
            ("kdr", "Key KDR", "Raw / corrected mismatch", ACCENT_GREEN),
        ]
        for idx, (key, title, subtitle, accent) in enumerate(specs):
            card = tk.Frame(self.frame, bg=CARD_BG, highlightthickness=1, highlightbackground=CARD_BORDER, bd=0)
            tk.Frame(card, bg=accent, height=2).pack(fill="x")
            inner = tk.Frame(card, bg=CARD_BG, padx=12, pady=8)
            inner.pack(fill="both", expand=True)
            tk.Label(inner, text=title.upper(), bg=CARD_BG, fg=TEXT_SOFT, font=ui_font("Arial", 8, "bold")).pack(anchor="w")
            if key == "kdr":
                value_row = tk.Frame(inner, bg=CARD_BG)
                value_row.pack(fill="x", pady=(5, 1))
                for metric_idx, (metric, label) in enumerate((("raw", "Raw"), ("cnn", "CNN"), ("cnnq", "CNN-Q"))):
                    value = tk.Label(
                        value_row,
                        text=f"{label}: --",
                        bg=CARD_BG,
                        fg=TEXT_MAIN,
                        font=ui_font("Consolas", 13, "bold"),
                        anchor="w",
                    )
                    value.pack(
                        side="left",
                        fill="x",
                        expand=True,
                        padx=(0, 14 if metric_idx < 2 else 0),
                    )
                    self.kdr_value_labels[metric] = value
            else:
                value = tk.Label(inner, text="--", bg=CARD_BG, fg=TEXT_MAIN, font=ui_font("Consolas", 13, "bold"))
                value.pack(anchor="w", pady=(5, 1))
                self.cards[key] = value
            tk.Label(inner, text=subtitle, bg=CARD_BG, fg=TEXT_MUTED, font=ui_font("Arial", 7)).pack(anchor="w")
            self.card_keys.append(key)
            self.card_widgets.append(card)
        self._layout_cards(6)
        self.frame.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        if event.width >= 1200:
            columns = 6
        elif event.width >= 900:
            columns = 3
        else:
            columns = 2
        self._layout_cards(columns)

    def _layout_cards(self, columns):
        if self.current_columns == columns:
            return
        self.current_columns = columns
        for idx in range(6):
            self.frame.columnconfigure(idx, weight=0, uniform="")
            self.frame.rowconfigure(idx, weight=0)
        for idx in range(columns):
            self.frame.columnconfigure(idx, weight=1, uniform="stats")

        row = 0
        col = 0
        for key, card in zip(self.card_keys, self.card_widgets):
            span = 2 if key == "kdr" else 1
            span = min(span, columns)
            if col + span > columns:
                row += 1
                col = 0
            top_pad = 0 if row == 0 else 8
            card.grid(
                row=row,
                column=col,
                columnspan=span,
                sticky="nsew",
                padx=5,
                pady=(top_pad, 0),
            )
            col += span
            if col >= columns:
                row += 1
                col = 0

    def update(self, serial, rssi, noise, epoch, latency, latency_ema, raw_kdr, cnn_kdr, cnnq_kdr):
        self.cards["serial"].config(text="--" if serial is None else str(serial))
        self.cards["rssi"].config(text="--" if rssi is None else f"{rssi:.1f}")
        self.cards["noise"].config(text="--" if noise is None else f"{noise:.1f}")
        # self.cards["epoch"].config(text="--" if epoch is None else str(epoch))
        if latency is None:
            self.cards["latency"].config(text="--")
        elif latency_ema is None:
            self.cards["latency"].config(text=f"{latency:.1f}")
        else:
            self.cards["latency"].config(text=f"{latency_ema:.1f} avg")
        self.kdr_value_labels["raw"].config(text=f"Raw: {'--' if raw_kdr is None else f'{raw_kdr:.1f}%'}")
        self.kdr_value_labels["cnn"].config(text=f"CNN: {'--' if cnn_kdr is None else f'{cnn_kdr:.1f}%'}")
        self.kdr_value_labels["cnnq"].config(text=f"CNN-Q: {'--' if cnnq_kdr is None else f'{cnnq_kdr:.1f}%'}")


class GSNDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.option_add("*Font", ui_font("Arial", 10))
        self.title("GSN Dashboard - PLKG Ground Station")
        self.geometry("1366x768")
        self.minsize(800, 500)
        self.configure(bg=APP_BG)

        self.state_obj = GSNState()
        self.log_queue = queue.Queue()
        self.log_lines = deque(maxlen=500)
        self.after_ids = []
        self.ui_paused = False
        self.backend_threads = []
        self.drag_panel = None
        self.drag_hover_host = None
        self.panel_hosts = {}
        self.panels = {}
        self.panel_visibility_vars = {}
        self.panel_menu_labels = {}
        self.layout_presets = []
        self.default_hosts = {}
        self.active_top_dropdown = None
        self.font_scale_var = tk.StringVar(value=self._font_scale_text())

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
        self.configure(bg=APP_BG)
        style.configure("Root.TFrame", background=APP_BG)
        style.configure("Card.TFrame", background=CARD_BG, relief="flat")
        style.configure("Shell.TFrame", background=SURFACE_BG, relief="flat")
        style.configure("Title.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=ui_font("Arial", 12, "bold"))
        style.configure("MetricTitle.TLabel", background=CARD_BG, foreground=TEXT_SOFT, font=ui_font("Arial", 9, "bold"))
        style.configure("Muted.TLabel", background=CARD_BG, foreground=TEXT_MUTED, font=ui_font("Arial", 10))
        style.configure("Header.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=ui_font("Arial", 16, "bold"))
        style.configure("SubHeader.TLabel", background=CARD_BG, foreground=TEXT_MUTED, font=ui_font("Arial", 9))
        style.configure("StatusBanner.TLabel", background=CARD_BG, foreground=ACCENT_BLUE, font=ui_font("Consolas", 10, "bold"))
        style.configure("SectionTitle.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=ui_font("Arial", 11, "bold"))
        style.configure("SectionMeta.TLabel", background=CARD_BG, foreground=TEXT_SOFT, font=ui_font("Arial", 8))
        style.configure("Badge.TLabel", background=CARD_BG_ALT, foreground=TEXT_MUTED, font=ui_font("Arial", 8, "bold"))
        style.configure("TButton", font=ui_font("Arial", 10, "bold"), padding=(9, 6))
        style.configure("StatusIdle.TLabel", background=CARD_BG, foreground=TEXT_MUTED, font=ui_font("Arial", 10, "bold"))
        style.configure("StatusWarn.TLabel", background=CARD_BG, foreground=ACCENT_AMBER, font=ui_font("Arial", 10, "bold"))
        style.configure("StatusGood.TLabel", background=CARD_BG, foreground=ACCENT_GREEN, font=ui_font("Arial", 10, "bold"))
        style.configure("StatusBad.TLabel", background=CARD_BG, foreground="#fb7185", font=ui_font("Arial", 10, "bold"))
        style.configure(
            "Panel.TCombobox",
            fieldbackground=CARD_BG_ALT,
            background=CARD_BG_ALT,
            foreground=TEXT_MAIN,
            font=ui_font("Arial", 9),
            arrowcolor=ACCENT_BLUE,
            bordercolor=CARD_BORDER,
            lightcolor=CARD_BORDER,
            darkcolor=CARD_BORDER,
            insertcolor=TEXT_MAIN,
            padding=4,
        )
        style.map(
            "Panel.TCombobox",
            fieldbackground=[("readonly", CARD_BG_ALT)],
            foreground=[("readonly", TEXT_MAIN)],
            selectbackground=[("readonly", CARD_BG_ALT)],
            selectforeground=[("readonly", TEXT_MAIN)],
        )

    def _font_scale_text(self):
        return f"{int(round(GUI_FONT_SCALE * 100))}%"

    def change_font_scale(self, delta):
        self.set_font_scale(GUI_FONT_SCALE + delta)

    def set_font_scale(self, value):
        old_scale, new_scale = set_gui_font_scale(value)
        if abs(old_scale - new_scale) < 1e-6:
            return

        self.option_add("*Font", ui_font("Arial", 10))
        self.font_scale_var.set(self._font_scale_text())
        self._configure_style()
        self._apply_font_scale(self, old_scale)
        for panel in self.panels.values():
            panel.render()

    def _apply_font_scale(self, widget, old_scale):
        self._scale_widget_font(widget, old_scale)
        for child in widget.winfo_children():
            self._apply_font_scale(child, old_scale)

    def _scale_widget_font(self, widget, old_scale):
        try:
            font_value = widget.cget("font")
        except tk.TclError:
            return
        if not font_value:
            return

        try:
            base_font = getattr(widget, "_plkg_base_font", None)
            if base_font is None:
                font_obj = tkfont.Font(root=self, font=font_value)
                actual = font_obj.actual()
                size = abs(int(actual.get("size", 10)))
                styles = []
                if actual.get("weight") == "bold":
                    styles.append("bold")
                if actual.get("slant") == "italic":
                    styles.append("italic")
                base_font = (
                    actual.get("family", "Arial"),
                    max(1.0, size / max(old_scale, 0.01)),
                    tuple(styles),
                )
                widget._plkg_base_font = base_font

            family, base_size, styles = base_font
            widget.configure(font=(family, ui_font_size(base_size), *styles))
        except Exception:
            return

    def _build_layout(self):
        root = ttk.Frame(self, style="Root.TFrame", padding=12)
        root.pack(fill="both", expand=True)
        self.root_frame = root

        topbar = tk.Frame(root, bg=CARD_BG, highlightthickness=1, highlightbackground=CARD_BORDER, bd=0)
        topbar.pack(fill="x", pady=(0, 10))
        tk.Frame(topbar, bg=ACCENT_BLUE, height=3).pack(fill="x")
        top_inner = tk.Frame(topbar, bg=CARD_BG, padx=14, pady=8)
        top_inner.pack(fill="x")

        title_block = tk.Frame(top_inner, bg=CARD_BG)
        title_block.pack(side="left", fill="x", expand=True)
        ttk.Label(title_block, text="GSN PLKG Dashboard", style="Header.TLabel").pack(anchor="w")
        tk.Label(
            title_block,
            text="Ground Station / PLKG Live Ops",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=ui_font("Arial", 9),
        ).pack(anchor="w", pady=(2, 0))

        status_shell = tk.Frame(
            top_inner,
            bg=CARD_BG_ALT,
            highlightthickness=1,
            highlightbackground=CARD_BORDER_SOFT,
            bd=0,
            padx=11,
            pady=6,
        )
        status_shell.pack(side="left", padx=(12, 10))
        tk.Label(status_shell, text="BACKEND", bg=CARD_BG_ALT, fg=TEXT_SOFT, font=ui_font("Arial", 8, "bold")).pack(anchor="e")
        self.status_banner = tk.Label(status_shell, text="OFF", bg=CARD_BG_ALT, fg=TEXT_MUTED, font=ui_font("Consolas", 9, "bold"))
        self.status_banner.pack(anchor="e", pady=(3, 0))

        controls = tk.Frame(top_inner, bg=CARD_BG)
        controls.pack(side="right")
        self.start_btn = ttk.Button(controls, text="Start", command=self.start_backend)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.encrypt_btn = ttk.Button(controls, text="Encrypt ON", command=self.toggle_video_encryption)
        self.encrypt_btn.pack(side="left", padx=(0, 6))
        self.pause_btn = ttk.Button(controls, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=(0, 6))
        self.clear_btn = ttk.Button(controls, text="Clear", command=self.clear_log)
        self.clear_btn.pack(side="left", padx=(0, 6))
        self.reset_layout_btn = ttk.Button(controls, text="Reset", command=self.reset_layout)
        self.reset_layout_btn.pack(side="left", padx=(0, 9))
        tk.Frame(controls, bg=CARD_BORDER_SOFT, width=1, height=26).pack(side="left", padx=(0, 9), pady=2)
        self.font_down_btn = ttk.Button(controls, text="A-", command=lambda: self.change_font_scale(-0.05), width=3)
        self.font_down_btn.pack(side="left", padx=(0, 4))
        self.font_scale_label = tk.Label(
            controls,
            textvariable=self.font_scale_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=ui_font("Consolas", 9, "bold"),
            width=5,
        )
        self.font_scale_label.pack(side="left", padx=(0, 4))
        self.font_up_btn = ttk.Button(controls, text="A+", command=lambda: self.change_font_scale(0.05), width=3)
        self.font_up_btn.pack(side="left", padx=(0, 9))
        tk.Frame(controls, bg=CARD_BORDER_SOFT, width=1, height=26).pack(side="left", padx=(0, 9), pady=2)
        self.panel_menu_button = ttk.Button(controls, text="Modules", command=self._toggle_panel_dropdown)
        self.panel_menu_button.pack(side="left", padx=(0, 6))
        self.layout_menu_button = ttk.Button(controls, text="Presets", command=self._toggle_layout_dropdown)
        self.layout_menu_button.pack(side="left")
        self.top_dropdown_shell = tk.Frame(topbar, bg=CARD_BG, padx=14)

        self.stats_strip = StatsStrip(root)

        workspace_shell = tk.Frame(root, bg=CARD_BG, highlightthickness=1, highlightbackground=CARD_BORDER, bd=0)
        workspace_shell.pack(fill="both", expand=True, pady=(0, 10))
        tk.Frame(workspace_shell, bg=ACCENT_VIOLET, height=3).pack(fill="x")
        workspace_head = tk.Frame(workspace_shell, bg=CARD_BG, padx=14, pady=8)
        workspace_head.pack(fill="x")
        ttk.Label(workspace_head, text="Live Workspace", style="SectionTitle.TLabel").pack(side="left")
        ttk.Label(workspace_head, text="Video first / telemetry right / analysis below", style="SectionMeta.TLabel").pack(side="right")
        workspace = ttk.PanedWindow(workspace_shell, orient="vertical")
        workspace.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.workspace_panes = {"workspace": workspace}
        main_row = ttk.PanedWindow(workspace, orient="horizontal")
        left_stack = ttk.PanedWindow(main_row, orient="vertical")
        right_stack = ttk.PanedWindow(main_row, orient="vertical")
        bottom_row = ttk.PanedWindow(workspace, orient="horizontal")
        footer = ttk.Frame(workspace, style="Root.TFrame")
        self.workspace_panes["main_row"] = main_row
        self.workspace_panes["left_stack"] = left_stack
        self.workspace_panes["right_stack"] = right_stack
        self.workspace_panes["bottom_row"] = bottom_row
        workspace.add(main_row, weight=7)
        workspace.add(bottom_row, weight=2)
        workspace.add(footer, weight=0)

        self.panel_hosts["primary"] = PanelHost(left_stack, "primary", "Empty slot")
        # self.panel_hosts["auxiliary"] = PanelHost(left_stack, "auxiliary", "Empty slot")
        self.panel_hosts["secondary"] = PanelHost(right_stack, "secondary", "Empty slot")
        # self.panel_hosts["tertiary"] = PanelHost(right_stack, "tertiary", "Empty slot")
        self.panel_hosts["analytics_left"] = PanelHost(bottom_row, "analytics_left", "Empty slot")
        self.panel_hosts["analytics_right"] = PanelHost(bottom_row, "analytics_right", "Empty slot")
        self.panel_hosts["footer"] = PanelHost(footer, "footer", "Empty slot")

        left_stack.add(self.panel_hosts["primary"], weight=6)
        # left_stack.add(self.panel_hosts["auxiliary"], weight=1)
        right_stack.add(self.panel_hosts["secondary"], weight=1)
        # right_stack.add(self.panel_hosts["tertiary"], weight=1)
        main_row.add(left_stack, weight=6)
        main_row.add(right_stack, weight=2)
        bottom_row.add(self.panel_hosts["analytics_left"], weight=1)
        bottom_row.add(self.panel_hosts["analytics_right"], weight=1)
        self.panel_hosts["footer"].pack(fill="both", expand=True)

        self.default_hosts = {
            "media_main": self.panel_hosts["primary"],
            # "control_panel": self.panel_hosts["auxiliary"],
            "text_main": self.panel_hosts["analytics_left"],
            "text_aux": self.panel_hosts["footer"],
            "chart_main": self.panel_hosts["secondary"],
            # "chart_aux": self.panel_hosts["tertiary"],
            "text_log": self.panel_hosts["analytics_right"],
        }
        self._init_panels()
        self._init_panel_menu()
        self._init_layout_menu()
        self.after(0, self.apply_auto_layout_preset)

    def _init_panels(self):
        self.panels = {
            "media_main": VideoModulePanel(self, "media_main", "Video Stream", "video"),
            "control_panel": ControlModulePanel(self, "control_panel", "System State", "session"),
            "text_main": TextModulePanel(self, "text_main", "Demo Key Snapshot", "demo_keys"),
            "text_aux": TextModulePanel(self, "text_aux", "Epoch History", "epoch_history"),
            "chart_main": ChartModulePanel(self, "chart_main", "Data Chart", "live_csi"),
            "chart_aux": ChartModulePanel(self, "chart_aux", "Live CSI", "live_csi"),
            "text_log": TextModulePanel(self, "text_log", "System Log", "epoch_history"),
        }
        self.default_visible_panels = {
            "media_main": True,
            "control_panel": False,
            "text_main": True,
            "text_aux": False,
            "chart_main": True,
            "chart_aux": False,
            "text_log": True,
        }
        for key, panel in self.panels.items():
            panel.visible = self.default_visible_panels.get(key, True)
            if panel.visible:
                panel.show()

    def _init_panel_menu(self):
        self.panel_menu_labels = {
            "media_main": "Video Stream",
            "control_panel": "System State",
            "text_main": "Demo Key Snapshot",
            "text_aux": "Epoch History",
            "chart_main": "Data Chart",
            "chart_aux": "Aux Chart",
            "text_log": "System Log",
        }
        for key in self.panel_menu_labels:
            var = tk.BooleanVar(value=self.panels[key].visible)
            self.panel_visibility_vars[key] = var

    def _init_layout_menu(self):
        self.layout_presets = [
            ("Auto Detect", None),
            ("Balanced", "balanced"),
            ("Wide Video", "wide_video"),
            ("Focus Video", "focus_video"),
            ("Analysis", "analysis"),
        ]

    def _toggle_panel_dropdown(self):
        if self.active_top_dropdown == "modules" and self.top_dropdown_shell.winfo_ismapped():
            self._hide_top_dropdown()
            return
        self._show_panel_dropdown()

    def _toggle_layout_dropdown(self):
        if self.active_top_dropdown == "presets" and self.top_dropdown_shell.winfo_ismapped():
            self._hide_top_dropdown()
            return
        self._show_layout_dropdown()

    def _prepare_top_dropdown(self, dropdown_name):
        for child in self.top_dropdown_shell.winfo_children():
            child.destroy()
        self.active_top_dropdown = dropdown_name
        if not self.top_dropdown_shell.winfo_ismapped():
            self.top_dropdown_shell.pack(fill="x", pady=(0, 9))
        container = tk.Frame(
            self.top_dropdown_shell,
            bg=CARD_BG_ALT,
            highlightthickness=1,
            highlightbackground=CARD_BORDER_SOFT,
            bd=0,
            padx=8,
            pady=6,
        )
        container.pack(side="right")
        return container

    def _hide_top_dropdown(self):
        for child in self.top_dropdown_shell.winfo_children():
            child.destroy()
        self.top_dropdown_shell.pack_forget()
        self.active_top_dropdown = None

    def _show_panel_dropdown(self):
        container = self._prepare_top_dropdown("modules")
        for idx, (key, label) in enumerate(self.panel_menu_labels.items()):
            var = self.panel_visibility_vars[key]
            var.set(self.panels[key].visible)
            check = tk.Checkbutton(
                container,
                text=label,
                variable=var,
                command=lambda k=key: self.toggle_panel(k),
                bg=CARD_BG_ALT,
                fg=TEXT_MAIN,
                activebackground=CARD_BG_ALT,
                activeforeground=TEXT_MAIN,
                selectcolor=CARD_BG_ALT,
                font=ui_font("Arial", 9, "bold"),
                relief="flat",
                bd=0,
                highlightthickness=0,
                padx=8,
                pady=4,
                anchor="w",
            )
            check.grid(row=idx // 4, column=idx % 4, sticky="w", padx=4, pady=2)

    def _show_layout_dropdown(self):
        container = self._prepare_top_dropdown("presets")
        for idx, (label, preset) in enumerate(self.layout_presets):
            button = tk.Button(
                container,
                text=label,
                command=lambda p=preset: self._select_layout_preset(p),
                bg=CARD_BG_ALT,
                fg=TEXT_MAIN,
                activebackground=CARD_BORDER_SOFT,
                activeforeground=TEXT_MAIN,
                font=ui_font("Arial", 9, "bold"),
                relief="flat",
                bd=0,
                highlightthickness=0,
                padx=10,
                pady=5,
            )
            button.grid(row=0, column=idx, sticky="ew", padx=3, pady=2)

    def _select_layout_preset(self, preset):
        self._hide_top_dropdown()
        if preset is None:
            self.apply_auto_layout_preset()
        else:
            self.apply_layout_preset(preset)

    def bind_panel_drag(self, panel, widget):
        widget.bind("<ButtonPress-1>", lambda event, p=panel: self.start_panel_drag(p))
        widget.bind("<B1-Motion>", self.update_panel_drag)
        widget.bind("<ButtonRelease-1>", self.end_panel_drag)

    def start_panel_drag(self, panel):
        self.drag_panel = panel
        self.configure(cursor="fleur")

    def update_panel_drag(self, event):
        if self.drag_panel is None:
            return
        host = self.find_host_from_pointer(event.x_root, event.y_root)
        if host is self.drag_hover_host:
            return
        if self.drag_hover_host is not None:
            self.drag_hover_host.set_drop_highlight(False)
        self.drag_hover_host = host
        if host is not None:
            host.set_drop_highlight(True)

    def end_panel_drag(self, event):
        if self.drag_panel is None:
            return
        host = self.find_host_from_pointer(event.x_root, event.y_root)
        if self.drag_hover_host is not None:
            self.drag_hover_host.set_drop_highlight(False)
        self.configure(cursor="")
        if host is not None and host is not self.drag_panel.host:
            self.drag_panel.mount(host)
        self.drag_panel = None
        self.drag_hover_host = None

    def find_host_from_pointer(self, x_root, y_root):
        widget = self.winfo_containing(x_root, y_root)
        while widget is not None:
            host = getattr(widget, "_panel_host_ref", None)
            if host is not None:
                return host
            widget = widget.master
        return None

    def toggle_panel(self, key):
        panel = self.panels[key]
        if self.panel_visibility_vars[key].get():
            panel.show()
        else:
            panel.hide()

    def reset_layout(self):
        for key, panel in self.panels.items():
            should_show = self.default_visible_panels.get(key, True)
            if isinstance(panel, ModulePanel):
                panel.content_key = panel.default_content
                if getattr(panel, "content_var", None) is not None:
                    panel.content_var.set(panel.content_options[panel.default_content])
            self.panel_visibility_vars[key].set(should_show)
            if should_show:
                panel.show()
                panel.mount(self.default_hosts[key])
            else:
                panel.hide()
        self.apply_auto_layout_preset()

    def apply_layout_preset(self, preset):
        self.update_idletasks()
        self._position_panes(preset)

    def _auto_layout_preset_name(self):
        screen_w = max(self.winfo_screenwidth(), 1)
        screen_h = max(self.winfo_screenheight(), 1)
        return "focus_video" if screen_h > screen_w else "balanced"

    def apply_auto_layout_preset(self):
        self.apply_layout_preset(self._auto_layout_preset_name())

    def _safe_sashpos(self, pane, index, value):
        try:
            pane.sashpos(index, int(value))
        except Exception:
            pass

    def _position_panes(self, preset):
        workspace = self.workspace_panes["workspace"]
        main_row = self.workspace_panes["main_row"]
        left_stack = self.workspace_panes["left_stack"]
        right_stack = self.workspace_panes["right_stack"]
        bottom_row = self.workspace_panes["bottom_row"]
        visible_by_host = {
            name: any(panel.visible and panel.host is host for panel in self.panels.values())
            for name, host in self.panel_hosts.items()
        }
        control_visible = visible_by_host.get("auxiliary", False)
        right_lower_visible = visible_by_host.get("tertiary", False)
        bottom_visible = visible_by_host.get("analytics_left", False) or visible_by_host.get("analytics_right", False)
        footer_visible = visible_by_host.get("footer", False)

        workspace_w = max(workspace.winfo_width(), 1)
        workspace_h = max(workspace.winfo_height(), 1)
        main_h = workspace_h
        side_w = workspace_w
        collapsed_bottom_h = 18

        if preset == "wide_video":
            main_split = workspace_h * (0.72 if bottom_visible else 0.96)
            bottom_split = main_split + (workspace_h * 0.25 if bottom_visible else collapsed_bottom_h)
            self._safe_sashpos(workspace, 0, main_split)
            self._safe_sashpos(workspace, 1, min(workspace_h * (0.94 if footer_visible else 0.975), bottom_split))
            self._safe_sashpos(main_row, 0, workspace_w * 0.64)
            self._safe_sashpos(left_stack, 0, main_h * (0.84 if control_visible else 0.985))
            self._safe_sashpos(right_stack, 0, main_h * (0.5 if right_lower_visible else 0.97))
            self._safe_sashpos(bottom_row, 0, side_w * 0.5)
            return

        if preset == "focus_video":
            main_split = workspace_h * (0.78 if bottom_visible else 0.97)
            bottom_split = main_split + (workspace_h * 0.18 if bottom_visible else collapsed_bottom_h)
            self._safe_sashpos(workspace, 0, main_split)
            self._safe_sashpos(workspace, 1, min(workspace_h * 0.985, bottom_split))
            self._safe_sashpos(main_row, 0, workspace_w * 0.70)
            self._safe_sashpos(left_stack, 0, main_h * (0.92 if control_visible else 0.985))
            self._safe_sashpos(right_stack, 0, main_h * (0.52 if right_lower_visible else 0.985))
            self._safe_sashpos(bottom_row, 0, side_w * 0.5)
            return

        if preset == "analysis":
            self._safe_sashpos(workspace, 0, workspace_h * 0.57)
            self._safe_sashpos(workspace, 1, workspace_h * (0.96 if bottom_visible else 0.59))
            self._safe_sashpos(main_row, 0, workspace_w * 0.52)
            self._safe_sashpos(left_stack, 0, main_h * (0.66 if control_visible else 0.97))
            self._safe_sashpos(right_stack, 0, main_h * (0.5 if right_lower_visible else 0.97))
            self._safe_sashpos(bottom_row, 0, side_w * 0.52)
            return

        main_split = workspace_h * (0.72 if bottom_visible else 0.96)
        bottom_split = main_split + (workspace_h * 0.24 if bottom_visible else collapsed_bottom_h)
        self._safe_sashpos(workspace, 0, main_split)
        self._safe_sashpos(workspace, 1, min(workspace_h * (0.94 if footer_visible else 0.975), bottom_split))
        self._safe_sashpos(main_row, 0, workspace_w * 0.64)
        self._safe_sashpos(left_stack, 0, main_h * (0.74 if control_visible else 0.985))
        self._safe_sashpos(right_stack, 0, main_h * (0.50 if right_lower_visible else 0.97))
        self._safe_sashpos(bottom_row, 0, side_w * 0.5)

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}] {message}")

    def clear_log(self):
        self.log_lines.clear()
        for panel in self.panels.values():
            if isinstance(panel, TextModulePanel) and panel.content_key == "log":
                panel.render()

    def _send_uav_control(self, message: bytes, target_ip=None):
        target = target_ip or "255.255.255.255"
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if target == "255.255.255.255":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            for _ in range(3):
                sock.sendto(message, (target, UAV_CONTROL_PORT))
                time.sleep(0.03)
        finally:
            sock.close()

    def toggle_video_encryption(self):
        with self.state_obj.lock:
            enabled = not self.state_obj.video_encryption_enabled
            self.state_obj.video_encryption_enabled = enabled
            target_ip = self.state_obj.latest_uav_ip
            self.state_obj.latest_eve_video_bgr = None
            self.state_obj.latest_eve_video_time = None
            self.state_obj.latest_eve_video_encrypted = None
            self.state_obj.latest_eve_video_decrypted = None

        command = f"VIDEO_ENCRYPTION {'1' if enabled else '0'}".encode("ascii")
        try:
            self._send_uav_control(command, target_ip=target_ip)
            mode = "ENCRYPTED" if enabled else "PLAINTEXT"
            target = target_ip or "broadcast"
            self.log(f"Requested UAV video mode: {mode} ({target}).")
        except Exception as e:
            self.log(f"Failed to send video encryption toggle: {e}")
        self._sync_encrypt_button()

    def _send_key_ack(self, uav_ip, epoch, serial_token, confirm):
        command = f"KEY_ACK {int(epoch)} {serial_token} {confirm}".encode("ascii")
        self._send_uav_control(command, target_ip=uav_ip)

    def _sync_encrypt_button(self):
        with self.state_obj.lock:
            enabled = self.state_obj.video_encryption_enabled
        self.encrypt_btn.config(text="Encrypt ON" if enabled else "Encrypt OFF")

    def toggle_pause(self):
        self.ui_paused = not self.ui_paused
        self.pause_btn.config(text="Resume" if self.ui_paused else "Pause")
        self.status_banner.config(
            text="Paused" if self.ui_paused else "Running",
            fg=ACCENT_AMBER if self.ui_paused else ACCENT_GREEN,
        )

    def start_backend(self):
        with self.state_obj.lock:
            if self.state_obj.started:
                self.log("Backend already started.")
                return
            self.state_obj.started = True
        self.start_btn.state(["disabled"])
        self.status_banner.config(text="Starting", fg=ACCENT_AMBER)
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
        t2 = threading.Thread(target=self._bch_worker, daemon=True)
        t1.start()
        t2.start()
        self.backend_threads.extend([t1, t2])
        if DEMO_TELEMETRY_ENABLED:
            t3 = threading.Thread(target=self._demo_telemetry_worker, daemon=True)
            t3.start()
            self.backend_threads.append(t3)
        elif DEMO_TELEMETRY_IMPORT_ERROR is not None:
            self.log(f"Demo telemetry disabled: {DEMO_TELEMETRY_IMPORT_ERROR}")

        try:
            rx = GSNReceiver(
                get_aes_key=self._get_key,
                on_frame=self._handle_frame,
                on_eve_frame=self._handle_eve_frame,
                get_eve_aes_key=self._get_eve_key,
            )
            rx.start()
            with self.state_obj.lock:
                self.state_obj.receiver_started = True
            self.log("GSNReceiver started on UDP/5005.")
        except Exception as e:
            self.log(f"Failed to start GSNReceiver: {e}")

    def _keygen_worker(self):
        try:
            # watcher = CSISerialWatcher("/dev/ttyUSB0", 115200)
            watcher = CSISerialWatcher(GSN_CSI_PORT, CSI_BAUD)  # macOS
            # watcher = CSISerialWatcher("COM3", 115200)  # Windows
            watcher.start()
            with self.state_obj.lock:
                self.state_obj.model_loaded = True
                self.state_obj.serial_connected = True
            self.log("GSN local quantizer ready. CNN/CNN-Q correction runs on UAV.")
        except Exception as e:
            self.log(f"Keygen init failed: {e}")
            return

        eve_watcher = None
        try:
            eve_watcher = CSISerialWatcher(EVE_CSI_PORT, CSI_BAUD, endpoint_type="EVE", device="EVE")
            eve_watcher.start()
            with self.state_obj.lock:
                self.state_obj.eve_serial_connected = True
            self.log(f"EVE CSI watcher started on {EVE_CSI_PORT}.")
        except Exception as e:
            self.log(f"EVE CSI watcher failed on {EVE_CSI_PORT}: {e}")

        last_serial = None
        last_eve_serial = None
        next_eve_key_update = time.monotonic() + random.uniform(0.0, next_eve_key_update_delay())
        while True:
            try:
                if eve_watcher is not None:
                    eve = eve_watcher.snapshot().get("EVE")
                    if eve:
                        eve_serial = eve.get("serial")
                        if eve_serial != last_eve_serial:
                            last_eve_serial = eve_serial
                            eve_csi = eve.get("csi")
                            eve_raw = None
                            eve_aes = None
                            eve_key_error = None
                            promote_eve_key = False
                            next_eve_delay = None
                            next_eve_update_wall = None
                            if eve_csi is not None:
                                eve_csi = np.asarray(eve_csi, dtype=np.float32)
                                if eve_csi.ndim == 1 and len(eve_csi) >= 10:
                                    try:
                                        eve_raw, eve_aes = generate_key(eve_csi)
                                        now_mono = time.monotonic()
                                        if now_mono >= next_eve_key_update:
                                            promote_eve_key = True
                                            next_eve_delay = next_eve_key_update_delay()
                                            next_eve_key_update = now_mono + next_eve_delay
                                            next_eve_update_wall = time.time() + next_eve_delay
                                    except Exception as exc:
                                        eve_key_error = str(exc)
                            with self.state_obj.lock:
                                self.state_obj.latest_eve_serial = eve_serial
                                self.state_obj.latest_eve_rssi = eve.get("rssi")
                                self.state_obj.latest_eve_noise = eve.get("noise")
                                self.state_obj.latest_eve_mac = eve.get("mac")
                                self.state_obj.latest_eve_csi_time = eve.get("time")
                                self.state_obj.latest_eve_csi = None if eve_csi is None else eve_csi.copy()
                                if eve_raw is not None and eve_aes is not None and eve_serial is not None:
                                    self.state_obj.latest_eve_raw = eve_raw
                                    self.state_obj.eve_raw_by_serial[eve_serial] = eve_raw
                                    self.state_obj.eve_aes_by_serial[eve_serial] = eve_aes
                                    self._trim_dict_locked(self.state_obj.eve_raw_by_serial, RAW_HISTORY_LIMIT)
                                    self._trim_dict_locked(self.state_obj.eve_aes_by_serial, RAW_HISTORY_LIMIT)
                                    if promote_eve_key:
                                        self.state_obj.active_eve_raw = eve_raw
                                        self.state_obj.active_eve_aes_key = eve_aes
                                        self.state_obj.active_eve_serial = eve_serial
                                        self.state_obj.active_eve_key_time = time.time()
                                        self.state_obj.active_eve_next_update_time = next_eve_update_wall
                                        self.state_obj.latest_eve_key_status = (
                                            f"Active EVE key serial {eve_serial}; "
                                            f"next independent update in {next_eve_delay:.1f}s."
                                        )
                                    elif self.state_obj.active_eve_aes_key is None:
                                        remaining = max(0.0, next_eve_key_update - time.monotonic())
                                        self.state_obj.latest_eve_key_status = (
                                            f"EVE observing serial {eve_serial}; first active key in {remaining:.1f}s."
                                        )
                                    else:
                                        remaining = max(0.0, next_eve_key_update - time.monotonic())
                                        self.state_obj.latest_eve_key_status = (
                                            f"EVE observing serial {eve_serial}; active serial "
                                            f"{self.state_obj.active_eve_serial}; next independent update in {remaining:.1f}s."
                                        )
                                elif eve_key_error:
                                    self.state_obj.latest_eve_key_status = f"EVE keygen failed: {eve_key_error}"
                                if eve.get("rssi") is not None:
                                    self.state_obj.eve_rssi_hist.append(eve.get("rssi"))
                                if eve.get("noise") is not None:
                                    self.state_obj.eve_noise_hist.append(eve.get("noise"))
                            if promote_eve_key:
                                self.log(
                                    f"EVE independent key update: active serial={eve_serial}; "
                                    f"next update in {next_eve_delay:.1f}s."
                                )
                            if isinstance(eve_serial, int) and eve_serial % 20 == 0:
                                self.log(
                                    f"Read EVE CSI seq {eve_serial} "
                                    f"rssi={eve.get('rssi', '--')} noise={eve.get('noise', '--')} "
                                    f"key={'ready' if eve_raw is not None else '--'}."
                                )

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

                raw, _ = generate_key(csi)
                with self.state_obj.lock:
                    self.state_obj.gsn_raw = raw
                    self.state_obj.gsn_raw_by_serial[serial] = raw
                    self.state_obj.gsn_csi_by_serial[serial] = csi.copy()
                    if len(self.state_obj.gsn_raw_by_serial) > RAW_HISTORY_LIMIT:
                        overflow = len(self.state_obj.gsn_raw_by_serial) - RAW_HISTORY_LIMIT
                        oldest = sorted(self.state_obj.gsn_raw_by_serial)[:overflow]
                        for old_serial in oldest:
                            self.state_obj.gsn_raw_by_serial.pop(old_serial, None)
                            self.state_obj.gsn_csi_by_serial.pop(old_serial, None)
                    self.state_obj.latest_serial = serial
                    self.state_obj.latest_rssi = rssi
                    self.state_obj.latest_noise = noise
                    self.state_obj.latest_csi_time = csi_time
                    self.state_obj.latest_gsn_live_serial = serial
                    self.state_obj.latest_gsn_live_csi = csi.copy()
                    self.state_obj.latest_gsn_live_csi_time = csi_time
                    self.state_obj.rssi_hist.append(rssi)
                    self.state_obj.noise_hist.append(noise)
                if serial % 20 == 0:
                    self.log(f"Generated local raw key from serial {serial}.")
            except Exception as e:
                self.log(f"Keygen loop error: {e}")
                time.sleep(0.1)

    def _bch_worker(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            sock.bind(("0.0.0.0", 5007))
            with self.state_obj.lock:
                self.state_obj.bch_started = True
            self.log("BCH helper receiver started on UDP/5007.")
        except Exception as e:
            self.log(f"BCH receiver init failed: {e}")
            return

        last_epoch = -1
        rejected_helpers = set()
        seen_helpers = set()
        waiting_helpers = set()
        while True:
            try:
                data, addr = sock.recvfrom(1024*1024)
                with self.state_obj.lock:
                    self.state_obj.latest_uav_ip = addr[0]
                parts = data.decode(errors="ignore").split()
                if len(parts) != 5 or parts[0] != "R":
                    continue
                epoch = int(parts[1])
                serial_token = parts[2]
                serial_pair = parse_serial_pair(serial_token)
                serial = serial_pair[0]
                peer_serial = serial_pair[1] if len(serial_pair) > 1 else serial
                helper = parts[3]
                confirm = parts[4]
                helper_id = (epoch, serial_token, helper, confirm)
                if helper_id not in seen_helpers:
                    seen_helpers.add(helper_id)
                    self.log(
                        f"Received BCH helper epoch={epoch} serials={serial_pair_label(serial_pair)} "
                        f"from {addr[0]}:{addr[1]}."
                    )

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
                        self._clear_demo_session_locked()
                        self.state_obj.latest_frame_bgr = None
                        self.state_obj.latest_frame_time = None
                        self.state_obj.video_status = "Requested UAV CSI reset. Waiting for fresh epoch."
                        self.state_obj.video_status_level = "warn"
                    self.log(f"Sent RESET_CSI to UAV at {addr[0]}")
                    last_epoch = -1
                    rejected_helpers.clear()
                    waiting_helpers.clear()
                    continue

                if last_epoch >= 0 and epoch < last_epoch:
                    with self.state_obj.lock:
                        self.state_obj.keys_by_epoch.clear()
                        self.state_obj.last_epoch = None
                        self.state_obj.active_key = None
                        self._clear_demo_session_locked()
                        self.state_obj.latest_frame_bgr = None
                        self.state_obj.latest_frame_time = None
                        self.state_obj.video_status = "UAV reboot detected. Waiting for key resync and fresh video."
                        self.state_obj.video_status_level = "warn"
                    self.log(f"UAV key session reset detected: epoch {last_epoch}->{epoch}")
                    rejected_helpers.clear()
                    waiting_helpers.clear()
                last_epoch = epoch

                with self.state_obj.lock:
                    self.state_obj.epoch_serial_by_epoch[epoch] = serial_pair
                    self._trim_dict_locked(self.state_obj.epoch_serial_by_epoch, RAW_HISTORY_LIMIT)

                with self.state_obj.lock:
                    existing_meta = self.state_obj.key_meta_by_epoch.get(epoch)
                    already_ready = epoch in self.state_obj.keys_by_epoch and existing_meta == helper_id
                    stale_same_epoch = epoch in self.state_obj.keys_by_epoch and existing_meta != helper_id
                    if stale_same_epoch:
                        self.state_obj.keys_by_epoch.pop(epoch, None)
                        self.state_obj.key_meta_by_epoch.pop(epoch, None)
                        self.state_obj.gsn_corrected_by_epoch.pop(epoch, None)
                        if self.state_obj.last_epoch == epoch:
                            self.state_obj.last_epoch = None
                            self.state_obj.active_key = None
                            self.state_obj.latest_frame_bgr = None
                            self.state_obj.latest_frame_time = None
                            self.state_obj.video_status = (
                                f"Detected new helper for reused epoch={epoch}; recomputing key."
                            )
                            self.state_obj.video_status_level = "warn"
                if already_ready:
                    try:
                        self._send_key_ack(addr[0], epoch, serial_token, confirm)
                    except Exception as exc:
                        self.log(f"Failed to resend KEY_ACK for epoch={epoch}: {exc}")
                    continue
                if stale_same_epoch:
                    self.log(
                        f"Reused epoch={epoch} arrived with new helper/confirm; discarded stale local key and recomputing."
                    )
                if helper_id in rejected_helpers:
                    continue

                with self.state_obj.lock:
                    local_raw = self.state_obj.gsn_raw_by_serial.get(serial)
                    local_peer_raw = self.state_obj.gsn_raw_by_serial.get(peer_serial)
                    latest_serial = self.state_obj.latest_serial
                    known_count = len(self.state_obj.gsn_raw_by_serial)
                if local_raw is None or local_peer_raw is None:
                    missing = [
                        str(item)
                        for item, value in ((serial, local_raw), (peer_serial, local_peer_raw))
                        if value is None
                    ]
                    with self.state_obj.lock:
                        self.state_obj.video_status = (
                            f"Waiting for local CSI serials={serial_pair_label(serial_pair)} before BCH correction."
                        )
                        self.state_obj.video_status_level = "warn"
                    if helper_id not in waiting_helpers:
                        waiting_helpers.add(helper_id)
                        self.log(
                            f"Waiting for local CSI serials={serial_pair_label(serial_pair)} "
                            f"for epoch={epoch}; missing={','.join(missing)}. "
                            f"latest_local_serial={latest_serial if latest_serial is not None else '--'} "
                            f"cached={known_count}."
                        )
                    continue

                with self.state_obj.lock:
                    self.state_obj.epoch_serial_by_epoch[epoch] = serial_pair
                    self._trim_dict_locked(self.state_obj.epoch_serial_by_epoch, RAW_HISTORY_LIMIT)

                try:
                    corrected = bch_decode_key(local_raw, helper)
                except ValueError as e:
                    kdr_hints = []
                    with self.state_obj.lock:
                        uav_demo = self.state_obj.uav_demo_by_epoch.get(epoch)
                        if uav_demo and parse_serial_pair(uav_demo.get("serial_pair", (uav_demo.get("serial"),))) == serial_pair:
                            for label, key_name, local_value in (
                                ("raw", "uav_raw_key", local_raw),
                                ("raw2", "uav_raw_key_2", local_peer_raw),
                                ("cnn", "uav_cnn_key", local_raw),
                                ("cnnq", "uav_cnnq_key", local_raw),
                                ("active", "uav_corrected_key", local_raw),
                            ):
                                key_value = uav_demo.get(key_name)
                                if key_value:
                                    kdr_hints.append(
                                        f"{label}_kdr_hint={self._kdr(local_value, key_value) * 100.0:.2f}%"
                                    )
                    with self.state_obj.lock:
                        self.state_obj.video_status = (
                            f"BCH correction failed for epoch={epoch}. Waiting for next helper."
                        )
                        self.state_obj.video_status_level = "warn"
                        self._refresh_demo_snapshot_locked(epoch)
                    hint = "" if not kdr_hints else " " + " ".join(kdr_hints)
                    self.log(f"BCH correction failed for epoch={epoch}, serials={serial_pair_label(serial_pair)}:{hint} {e}")
                    continue
                aes = sha256.sha_byte(corrected)
                if not verify_key_confirm(aes, epoch, serial_token, helper, confirm):
                    rejected_helpers.add(helper_id)
                    with self.state_obj.lock:
                        self.state_obj.video_status = "Pending key confirmation failed. Keeping previous video key."
                        self.state_obj.video_status_level = "warn"
                    self.log(f"Key confirmation failed for epoch={epoch}, serials={serial_pair_label(serial_pair)}.")
                    continue

                raw_kdr = self._kdr(local_raw, corrected) * 100.0

                demo_snapshot = None
                with self.state_obj.lock:
                    self.state_obj.keys_by_epoch[epoch] = aes
                    self.state_obj.key_meta_by_epoch[epoch] = helper_id
                    self.state_obj.gsn_corrected_by_epoch[epoch] = corrected
                    self.state_obj.epoch_serial_by_epoch[epoch] = serial_pair
                    self._trim_dict_locked(self.state_obj.key_meta_by_epoch, RAW_HISTORY_LIMIT)
                    self._trim_dict_locked(self.state_obj.gsn_corrected_by_epoch, RAW_HISTORY_LIMIT)
                    self._trim_dict_locked(self.state_obj.epoch_serial_by_epoch, RAW_HISTORY_LIMIT)
                    demo_snapshot = self._refresh_demo_snapshot_locked(epoch)
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
                try:
                    self._send_key_ack(addr[0], epoch, serial_token, confirm)
                    self.log(f"Sent KEY_ACK epoch={epoch} serials={serial_pair_label(serial_pair)} to {addr[0]}.")
                except Exception as exc:
                    self.log(f"Failed to send KEY_ACK for epoch={epoch}: {exc}")
                if demo_snapshot:
                    self.log(
                        f"[KEY ACTIVE] epoch={epoch} serial={serial} "
                        f"serials={serial_pair_label(serial_pair)} "
                        f"raw_kdr={demo_snapshot['raw_kdr']:.2f}% "
                        f"cnn_kdr={fmt_pct(demo_snapshot.get('cnn_kdr'))} "
                        f"cnnq_kdr={fmt_pct(demo_snapshot.get('cnnq_kdr'))} "
                        f"correction_bits={raw_kdr:.2f}%"
                    )
                else:
                    self.log(f"[KEY ACTIVE] epoch={epoch} serials={serial_pair_label(serial_pair)} confirmed corrected_bits={raw_kdr:.2f}%")
            except Exception as e:
                self.log(f"BCH loop error: {e}")
                time.sleep(0.05)

    def _demo_telemetry_worker(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", DEMO_TELEMETRY_PORT))
            with self.state_obj.lock:
                self.state_obj.demo_telemetry_started = True
            self.log(f"Demo telemetry receiver started on UDP/{DEMO_TELEMETRY_PORT}.")
        except Exception as e:
            self.log(f"Demo telemetry receiver init failed: {e}")
            return

        raw_logged_epochs = set()
        joined_logged_epochs = set()
        while True:
            try:
                data, addr = sock.recvfrom(65535)
                payload = parse_telemetry_packet(data)
                if not payload:
                    continue
                with self.state_obj.lock:
                    self.state_obj.latest_uav_ip = addr[0]
                if payload.get("packet_type") == "live_csi":
                    live_csi = np.asarray(payload.get("uav_live_csi"), dtype=np.float32)
                    live_cnn_csi = payload.get("uav_live_cnn_csi")
                    if live_cnn_csi is not None:
                        live_cnn_csi = np.asarray(live_cnn_csi, dtype=np.float32)
                    with self.state_obj.lock:
                        self.state_obj.latest_uav_live_serial = payload.get("serial")
                        self.state_obj.latest_uav_live_csi = live_csi.copy()
                        self.state_obj.latest_uav_live_cnn_csi = (
                            None if live_cnn_csi is None else live_cnn_csi.copy()
                        )
                        self.state_obj.latest_uav_live_cnn_serial_pair = payload.get("cnn_serial_pair")
                        self.state_obj.latest_uav_live_epoch = payload.get("epoch")
                        self.state_obj.latest_uav_live_csi_time = payload.get("time", time.time())
                        if "uav_rssi" in payload and payload["uav_rssi"]:
                            uav_rssi_val = payload["uav_rssi"][0] if isinstance(payload["uav_rssi"], list) else payload["uav_rssi"]
                            if uav_rssi_val is not None:
                                self.state_obj.uav_rssi_hist.append(float(uav_rssi_val))
                    continue
                epoch = payload["epoch"]
                serial = payload["serial"]
                with self.state_obj.lock:
                    self.state_obj.uav_demo_by_epoch[epoch] = payload
                    self.state_obj.latest_uav_demo = dict(payload)
                    self.state_obj.latest_demo_telemetry_time = payload.get("time", time.time())
                    
                    # Append UAV RSSI to history if available
                    if "uav_rssi" in payload and payload["uav_rssi"]:
                        uav_rssi_val = payload["uav_rssi"][0] if isinstance(payload["uav_rssi"], list) else payload["uav_rssi"]
                        if uav_rssi_val is not None:
                            self.state_obj.uav_rssi_hist.append(float(uav_rssi_val))

                    self._trim_dict_locked(self.state_obj.uav_demo_by_epoch, RAW_HISTORY_LIMIT)
                    demo_snapshot = self._refresh_demo_snapshot_locked(epoch)
                if epoch not in raw_logged_epochs:
                    raw_logged_epochs.add(epoch)
                    self.log(
                        f"Received demo telemetry epoch={epoch} "
                        f"serials={serial_pair_label(payload.get('serial_pair', (serial,)))}."
                    )
                if demo_snapshot and epoch not in joined_logged_epochs:
                    joined_logged_epochs.add(epoch)
                    self.log(
                        f"Demo telemetry joined epoch={epoch} "
                        f"raw_kdr={demo_snapshot['raw_kdr']:.2f}% "
                        f"corrected_kdr={demo_snapshot['corrected_kdr']:.2f}%"
                    )
            except Exception as e:
                self.log(f"Demo telemetry loop error: {e}")
                time.sleep(0.05)

    @staticmethod
    def _trim_dict_locked(values, limit):
        if len(values) <= limit:
            return
        overflow = len(values) - limit
        for key in sorted(values)[:overflow]:
            values.pop(key, None)

    def _clear_demo_session_locked(self):
        self.state_obj.gsn_corrected_by_epoch.clear()
        self.state_obj.epoch_serial_by_epoch.clear()
        self.state_obj.key_meta_by_epoch.clear()
        self.state_obj.uav_demo_by_epoch.clear()
        self.state_obj.latest_uav_demo = None
        self.state_obj.latest_demo = None
        self.state_obj.latest_demo_telemetry_time = None
        self.state_obj.latest_uav_live_serial = None
        self.state_obj.latest_uav_live_csi = None
        self.state_obj.latest_uav_live_cnn_csi = None
        self.state_obj.latest_uav_live_cnn_serial_pair = None
        self.state_obj.latest_uav_live_epoch = None
        self.state_obj.latest_uav_live_csi_time = None
        self.state_obj.latest_gsn_live_serial = None
        self.state_obj.latest_gsn_live_csi = None
        self.state_obj.latest_gsn_live_csi_time = None
        self.state_obj.latest_demo_raw_kdr = None
        self.state_obj.latest_demo_cnn_kdr = None
        self.state_obj.latest_demo_cnnq_kdr = None
        self.state_obj.latest_eve_video_bgr = None
        self.state_obj.latest_eve_video_time = None
        self.state_obj.latest_eve_video_encrypted = None
        self.state_obj.latest_eve_video_decrypted = None
        self.state_obj.demo_raw_kdr_hist.clear()
        self.state_obj.demo_cnn_kdr_hist.clear()
        self.state_obj.demo_cnnq_kdr_hist.clear()
        self.state_obj.demo_hist_epochs.clear()

    def _refresh_demo_snapshot_locked(self, epoch):
        uav_demo = self.state_obj.uav_demo_by_epoch.get(epoch)
        gsn_corrected_key = self.state_obj.gsn_corrected_by_epoch.get(epoch)
        serial_pair = self.state_obj.epoch_serial_by_epoch.get(epoch)
        if not uav_demo or serial_pair is None:
            return None
        serial_pair = parse_serial_pair(serial_pair)
        if parse_serial_pair(uav_demo.get("serial_pair", (uav_demo.get("serial"),))) != serial_pair:
            return None
        serial = serial_pair[0]
        peer_serial = serial_pair[1] if len(serial_pair) > 1 else serial

        gsn_raw_key = self.state_obj.gsn_raw_by_serial.get(serial)
        gsn_raw_key_2 = self.state_obj.gsn_raw_by_serial.get(peer_serial)
        gsn_raw_csi = self.state_obj.gsn_csi_by_serial.get(serial)
        gsn_raw_csi_2 = self.state_obj.gsn_csi_by_serial.get(peer_serial)
        if gsn_raw_key is None or gsn_raw_key_2 is None or gsn_raw_csi is None or gsn_raw_csi_2 is None:
            return None

        raw_kdr = self._kdr(gsn_raw_key, uav_demo.get("uav_raw_key")) * 100.0
        cnn_kdr = self._kdr(gsn_raw_key, uav_demo.get("uav_cnn_key")) * 100.0 if uav_demo.get("uav_cnn_key") else None
        cnnq_kdr = self._kdr(gsn_raw_key, uav_demo.get("uav_cnnq_key")) * 100.0 if uav_demo.get("uav_cnnq_key") else None
        snapshot = {
            "epoch": epoch,
            "serial": serial,
            "serial_pair": serial_pair,
            "uav_rssi": uav_demo.get("uav_rssi"),
            "uav_raw_csi": list(uav_demo["uav_raw_csi"]),
            "uav_raw_csi_2": uav_demo.get("uav_raw_csi_2"),
            "uav_cnn_csi": uav_demo.get("uav_cnn_csi"),
            "gsn_raw_csi": np.asarray(gsn_raw_csi, dtype=np.float32).tolist(),
            "gsn_raw_csi_2": np.asarray(gsn_raw_csi_2, dtype=np.float32).tolist(),
            "uav_raw_key": uav_demo.get("uav_raw_key"),
            "uav_raw_key_2": uav_demo.get("uav_raw_key_2"),
            "gsn_raw_key": gsn_raw_key,
            "gsn_raw_key_2": gsn_raw_key_2,
            "uav_cnn_key": uav_demo.get("uav_cnn_key"),
            "uav_cnnq_key": uav_demo.get("uav_cnnq_key"),
            "uav_corrected_key": uav_demo.get("uav_corrected_key"),
            "gsn_corrected_key": gsn_corrected_key,
            "eve_csi": list(self.state_obj.latest_eve_csi) if self.state_obj.latest_eve_csi is not None else None,
            "raw_kdr": raw_kdr,
            "cnn_kdr": cnn_kdr,
            "cnnq_kdr": cnnq_kdr,
            "time": time.time(),
        }
        self.state_obj.latest_demo = snapshot
        self.state_obj.latest_demo_raw_kdr = raw_kdr
        self.state_obj.latest_demo_cnn_kdr = cnn_kdr
        self.state_obj.latest_demo_cnnq_kdr = cnnq_kdr

        if epoch not in self.state_obj.demo_hist_epochs:
            self.state_obj.demo_raw_kdr_hist.append(raw_kdr)
            if cnn_kdr is not None:
                self.state_obj.demo_cnn_kdr_hist.append(cnn_kdr)
            if cnnq_kdr is not None:
                self.state_obj.demo_cnnq_kdr_hist.append(cnnq_kdr)
            self.state_obj.demo_hist_epochs.add(epoch)
            if len(self.state_obj.demo_hist_epochs) > RAW_HISTORY_LIMIT:
                self.state_obj.demo_hist_epochs.clear()

        return snapshot

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

    def _handle_eve_frame(self, frame, encrypted, decrypted=False):
        if frame is None:
            return
        with self.state_obj.lock:
            self.state_obj.latest_eve_video_bgr = frame.copy()
            self.state_obj.latest_eve_video_time = time.time()
            self.state_obj.latest_eve_video_encrypted = bool(encrypted)
            self.state_obj.latest_eve_video_decrypted = bool(decrypted)

    def _get_key(self, epoch):
        with self.state_obj.lock:
            return self.state_obj.keys_by_epoch.get(epoch)

    def _get_eve_key(self, epoch):
        with self.state_obj.lock:
            return self.state_obj.active_eve_aes_key

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
                self.log_lines.append(msg)
        except queue.Empty:
            pass
        self.after_ids.append(self.after(200, self._drain_log_queue))

    def _refresh_ui(self):
        if not self.ui_paused:
            with self.state_obj.lock:
                serial = self.state_obj.latest_serial
                rssi = self.state_obj.latest_rssi
                noise = self.state_obj.latest_noise
                eve_serial = self.state_obj.latest_eve_serial
                eve_rssi = self.state_obj.latest_eve_rssi
                eve_noise = self.state_obj.latest_eve_noise
                eve_mac = self.state_obj.latest_eve_mac
                eve_csi = None if self.state_obj.latest_eve_csi is None else self.state_obj.latest_eve_csi.copy()
                eve_raw = self.state_obj.active_eve_raw or self.state_obj.latest_eve_raw
                eve_key_status = self.state_obj.latest_eve_key_status
                gsn_live_serial = self.state_obj.latest_gsn_live_serial
                gsn_live_csi = None if self.state_obj.latest_gsn_live_csi is None else self.state_obj.latest_gsn_live_csi.copy()
                uav_live_serial = self.state_obj.latest_uav_live_serial
                uav_live_csi = None if self.state_obj.latest_uav_live_csi is None else self.state_obj.latest_uav_live_csi.copy()
                uav_live_cnn_csi = (
                    None
                    if self.state_obj.latest_uav_live_cnn_csi is None
                    else self.state_obj.latest_uav_live_cnn_csi.copy()
                )
                uav_live_cnn_serial_pair = self.state_obj.latest_uav_live_cnn_serial_pair
                uav_live_epoch = self.state_obj.latest_uav_live_epoch
                epoch = self.state_obj.last_epoch
                latency = self.state_obj.latest_latency_ms
                latency_ema = self.state_obj.latest_latency_ema_ms
                frame_time = self.state_obj.latest_frame_time
                raw_kdr = self.state_obj.latest_kdr_raw
                corr_kdr = self.state_obj.latest_kdr_corr
                demo = None if self.state_obj.latest_demo is None else dict(self.state_obj.latest_demo)
                uav_demo = None if self.state_obj.latest_uav_demo is None else dict(self.state_obj.latest_uav_demo)
                demo_raw_kdr = self.state_obj.latest_demo_raw_kdr
                demo_cnn_kdr = self.state_obj.latest_demo_cnn_kdr
                demo_cnnq_kdr = self.state_obj.latest_demo_cnnq_kdr
                frame = None if self.state_obj.latest_frame_bgr is None else self.state_obj.latest_frame_bgr.copy()
                eve_video_frame = None if self.state_obj.latest_eve_video_bgr is None else self.state_obj.latest_eve_video_bgr.copy()
                eve_video_time = self.state_obj.latest_eve_video_time
                eve_video_encrypted = self.state_obj.latest_eve_video_encrypted
                eve_video_decrypted = self.state_obj.latest_eve_video_decrypted
                video_encryption_enabled = self.state_obj.video_encryption_enabled
                gsn_raw = self.state_obj.gsn_raw
                aes_key = self.state_obj.active_key
                video_status = self.state_obj.video_status
                video_status_level = self.state_obj.video_status_level
                latest_uav_ip = self.state_obj.latest_uav_ip
                keys_by_epoch = dict(self.state_obj.keys_by_epoch)
                kdr_raw_hist = list(self.state_obj.kdr_raw_hist)
                kdr_corr_hist = list(self.state_obj.kdr_corr_hist)
                demo_raw_kdr_hist = list(self.state_obj.demo_raw_kdr_hist)
                demo_cnn_kdr_hist = list(self.state_obj.demo_cnn_kdr_hist)
                demo_cnnq_kdr_hist = list(self.state_obj.demo_cnnq_kdr_hist)
                lat_hist = list(self.state_obj.latency_hist)
                lat_ema_hist = list(self.state_obj.latency_ema_hist)
                rssi_hist = list(self.state_obj.rssi_hist)
                uav_rssi_hist = list(self.state_obj.uav_rssi_hist)
                noise_hist = list(self.state_obj.noise_hist)
                eve_rssi_hist = list(self.state_obj.eve_rssi_hist)
                eve_noise_hist = list(self.state_obj.eve_noise_hist)
                running = self.state_obj.started
                model_loaded = self.state_obj.model_loaded
                serial_ok = self.state_obj.serial_connected
                eve_serial_ok = self.state_obj.eve_serial_connected
                rx_ok = self.state_obj.receiver_started
                bch_ok = self.state_obj.bch_started
                demo_ok = self.state_obj.demo_telemetry_started

            # strip_raw_kdr = demo_raw_kdr if demo_raw_kdr is not None else raw_kdr
            self.stats_strip.update(serial, rssi, noise, epoch, latency, latency_ema, demo_raw_kdr, demo_cnn_kdr, demo_cnnq_kdr)
            snapshot = {
                "serial": serial,
                "rssi": rssi,
                "noise": noise,
                "eve_serial": eve_serial,
                "eve_rssi": eve_rssi,
                "eve_noise": eve_noise,
                "eve_mac": eve_mac,
                "eve_csi": eve_csi,
                "eve_raw": eve_raw,
                "eve_key_status": eve_key_status,
                "gsn_live_serial": gsn_live_serial,
                "gsn_live_csi": gsn_live_csi,
                "uav_live_serial": uav_live_serial,
                "uav_live_csi": uav_live_csi,
                "uav_live_cnn_csi": uav_live_cnn_csi,
                "uav_live_cnn_serial_pair": uav_live_cnn_serial_pair,
                "uav_live_epoch": uav_live_epoch,
                "epoch": epoch,
                "latency": latency,
                "latency_ema": latency_ema,
                "frame_time": frame_time,
                "raw_kdr": raw_kdr,
                "corr_kdr": corr_kdr,
                "demo": demo,
                "uav_demo": uav_demo,
                "demo_raw_kdr": demo_raw_kdr,
                "demo_cnn_kdr": demo_cnn_kdr,
                "demo_cnnq_kdr": demo_cnnq_kdr,
                "uav_rssi_hist": uav_rssi_hist,
                "frame": frame,
                "eve_video_frame": eve_video_frame,
                "eve_video_time": eve_video_time,
                "eve_video_encrypted": eve_video_encrypted,
                "eve_video_decrypted": eve_video_decrypted,
                "video_encryption_enabled": video_encryption_enabled,
                "gsn_raw": gsn_raw,
                "aes_key": aes_key,
                "video_status": video_status,
                "video_status_level": video_status_level,
                "latest_uav_ip": latest_uav_ip,
                "keys_by_epoch": keys_by_epoch,
                "kdr_raw_hist": kdr_raw_hist,
                "kdr_corr_hist": kdr_corr_hist,
                "demo_raw_kdr_hist": demo_raw_kdr_hist,
                "demo_cnn_kdr_hist": demo_cnn_kdr_hist,
                "demo_cnnq_kdr_hist": demo_cnnq_kdr_hist,
                "lat_hist": lat_hist,
                "lat_ema_hist": lat_ema_hist,
                "rssi_hist": rssi_hist,
                "noise_hist": noise_hist,
                "eve_rssi_hist": eve_rssi_hist,
                "eve_noise_hist": eve_noise_hist,
                "started": running,
                "model_loaded": model_loaded,
                "serial_ok": serial_ok,
                "eve_serial_ok": eve_serial_ok,
                "rx_ok": rx_ok,
                "bch_ok": bch_ok,
                "demo_ok": demo_ok,
                "demo_enabled": DEMO_TELEMETRY_ENABLED,
            }
            for panel in self.panels.values():
                panel.update_snapshot(snapshot)
            self._sync_encrypt_button()
            if running:
                ready_items = [model_loaded, serial_ok, eve_serial_ok, rx_ok, bch_ok]
                if DEMO_TELEMETRY_ENABLED:
                    ready_items.append(demo_ok)
                ready_count = sum(1 for item in ready_items if item)
                ready_total = len(ready_items)
                self.status_banner.config(
                    text=f"ON {ready_count}/{ready_total}",
                    fg=ACCENT_GREEN if ready_count == ready_total else ACCENT_AMBER,
                )
            else:
                self.status_banner.config(text="OFF", fg=TEXT_MUTED)

        self.after_ids.append(self.after(33, self._refresh_ui))

    @staticmethod
    def video_hint_style(level):
        if level == "good":
            return "StatusGood.TLabel"
        if level == "warn":
            return "StatusWarn.TLabel"
        if level == "bad":
            return "StatusBad.TLabel"
        return "StatusIdle.TLabel"

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
