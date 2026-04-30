import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse

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
from bch_reconciliation import bch_decode_key
from gsn_key_generate import load_model, CSISerialWatcher, generate_key
from gsn_receiver import GSNReceiver

UAV_CONTROL_PORT = 5008
RAW_HISTORY_LIMIT = 512
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


@dataclass
class GSNState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    model_loaded: bool = False
    serial_connected: bool = False
    receiver_started: bool = False
    bch_started: bool = False

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
    noise_hist: deque = field(default_factory=lambda: deque(maxlen=100))
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
            font=("Arial", 10, "italic"),
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
        grip = tk.Label(titlebar, text="::", bg=TITLE_BG, fg=TEXT_SOFT, font=("Consolas", 11, "bold"))
        grip.pack(side="left", padx=(8, 4))
        label = tk.Label(titlebar, text=self.title, bg=TITLE_BG, fg=TEXT_MAIN, font=("Arial", 11, "bold"))
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
        grip = tk.Label(titlebar, text="::", bg=TITLE_BG, fg=TEXT_SOFT, font=("Consolas", 12, "bold"))
        grip.pack(side="left", padx=(0, 7))
        label = tk.Label(titlebar, text=self.title, bg=TITLE_BG, fg=TEXT_MAIN, font=("Arial", 11, "bold"))
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
            tk.Label(titlebar, text=self.content_options[self.content_key], bg=TITLE_BG, fg=TEXT_MUTED, font=("Arial", 9, "bold")).pack(side="right", padx=10)

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
    def __init__(self, dashboard, key="media_main", title="Decrypted Video", default_content="video"):
        super().__init__(dashboard, key, title, {"video": "Decrypted Video"}, default_content)
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
        self.video_label = tk.Label(self.viewport, text="Waiting for frames...", bg="#020617", fg="#cbd5e1")
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
            self.video_label.config(image="", text="Video stalled.\nWaiting for UAV stream recovery...")
            self.video_label.place(relx=0.5, rely=0.5, anchor="center")
            self.video_hint.config(text=video_status, style=self.dashboard.video_hint_style("bad"))
            return

        disp = frame.copy()
        if latency is not None:
            cv2.putText(disp, f"Latency={latency:.1f} ms", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
        self.video_hint.config(text=video_status, style=self.dashboard.video_hint_style(video_status_level))


class TextModulePanel(ModulePanel):
    TEXT_OPTIONS = {
        "key_status": "Key Status",
        "epoch_history": "Epoch History",
        "log": "System Log",
        "link_snapshot": "Link Snapshot",
    }

    def __init__(self, dashboard, key, title, default_content):
        super().__init__(dashboard, key, title, self.TEXT_OPTIONS, default_content)

    def build_body(self, parent):
        self.text = tk.Text(parent, wrap="word", bg="#020617", fg="#e2e8f0", insertbackground="#e2e8f0", relief="flat")
        self.text.pack(fill="both", expand=True, padx=10, pady=10)
        self.text.config(state="disabled")

    def render(self):
        if self.card is None:
            return

        serial = self.snapshot.get("serial")
        rssi = self.snapshot.get("rssi")
        noise = self.snapshot.get("noise")
        gsn_raw = self.snapshot.get("gsn_raw")
        epoch = self.snapshot.get("epoch")
        aes_key = self.snapshot.get("aes_key")
        keys_by_epoch = self.snapshot.get("keys_by_epoch", {})
        latency = self.snapshot.get("latency")
        latency_ema = self.snapshot.get("latency_ema")
        raw_kdr = self.snapshot.get("raw_kdr")
        corr_kdr = self.snapshot.get("corr_kdr")
        video_status = self.snapshot.get("video_status")

        if self.content_key == "key_status":
            lines = [
                f"serial       : {serial if serial is not None else '--'}",
                f"rssi/noise   : {('--' if rssi is None else f'{rssi:.1f}')}/{('--' if noise is None else f'{noise:.1f}')}",
                f"raw key      : {(gsn_raw[:64] + '...') if gsn_raw else '--'}",
                f"active epoch : {epoch if epoch is not None else '--'}",
                f"aes key      : {(aes_key.hex()[:48] + '...') if isinstance(aes_key, (bytes, bytearray)) else '--'}",
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
                f"latency raw   : {('--' if latency is None else f'{latency:.1f} ms')}",
                f"latency avg   : {('--' if latency_ema is None else f'{latency_ema:.1f} ms')}",
                f"corr bits     : {('--' if raw_kdr is None else f'{raw_kdr:.2f}%')}",
                f"post-check    : {('--' if corr_kdr is None else f'{corr_kdr:.2f}%')}",
                f"video status  : {video_status or '--'}",
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

    def render(self):
        if self.card is None:
            return

        self.ax.clear()
        self.ax.set_facecolor("#020617")
        self.ax.tick_params(colors="#cbd5e1")
        for spine in self.ax.spines.values():
            spine.set_color("#475569")

        if self.content_key == "latency":
            lat_hist = list(self.snapshot.get("lat_hist", []))
            lat_ema_hist = list(self.snapshot.get("lat_ema_hist", []))
            if lat_hist:
                x = list(range(len(lat_hist)))
                self.ax.plot(x, lat_hist, label="Latency raw", alpha=0.35)
            if lat_ema_hist:
                x_ema = list(range(len(lat_ema_hist)))
                self.ax.plot(x_ema, lat_ema_hist, label="Latency smoothed", linewidth=2.0)
            self.ax.set_title("Latency", color="#e5e7eb")
            if lat_hist or lat_ema_hist:
                self.ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb")

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
            self.ax.set_title("Correction (%)", color="#e5e7eb")
            if raw_hist or corr_hist:
                self.ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb")

        else:
            rssi_hist = list(self.snapshot.get("rssi_hist", []))
            noise_hist = list(self.snapshot.get("noise_hist", []))
            if rssi_hist:
                x = list(range(len(rssi_hist)))
                self.ax.plot(x, rssi_hist, label="RSSI", linewidth=2.0)
            if noise_hist:
                x2 = list(range(len(noise_hist)))
                self.ax.plot(x2, noise_hist, label="Noise", linewidth=1.6, alpha=0.9)
            self.ax.set_title("RSSI / Noise", color="#e5e7eb")
            if rssi_hist or noise_hist:
                self.ax.legend(facecolor="#111827", edgecolor="#475569", labelcolor="#e5e7eb")

        self.canvas.draw_idle()


class ControlModulePanel(ModulePanel):
    CONTROL_OPTIONS = {"session": "Runtime State"}

    def __init__(self, dashboard, key="control_panel", title="System State", default_content="session"):
        super().__init__(dashboard, key, title, self.CONTROL_OPTIONS, default_content)

    def build_body(self, parent):
        self.summary = tk.Text(
            parent,
            height=5,
            wrap="word",
            bg="#020617",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
        )
        self.summary.pack(fill="both", expand=True, padx=10, pady=10)
        self.summary.config(state="disabled")

    def render(self):
        if self.card is None:
            return

        lines = [
            f"backend      : {'ON' if self.snapshot.get('started') else 'OFF'}",
            f"model loaded : {'ON' if self.snapshot.get('model_loaded') else 'OFF'}",
            f"serial link  : {'ON' if self.snapshot.get('serial_ok') else 'OFF'}",
            f"video rx     : {'ON' if self.snapshot.get('rx_ok') else 'OFF'}",
            f"bch rx       : {'ON' if self.snapshot.get('bch_ok') else 'OFF'}",
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
        self.card_widgets = []
        self.current_columns = None
        specs = [
            ("serial", "Latest Serial", "Stream sequence", ACCENT_BLUE),
            ("rssi", "RSSI", "Radio strength", ACCENT_TEAL),
            ("noise", "Noise", "Channel floor", ACCENT_AMBER),
            ("epoch", "Active Epoch", "Confirmed key window", ACCENT_VIOLET),
            ("latency", "Latency (ms)", "End-to-end response", ACCENT_PINK),
            ("kdr", "Correction Bits", "BCH correction ratio", ACCENT_GREEN),
        ]
        for idx, (key, title, subtitle, accent) in enumerate(specs):
            card = tk.Frame(self.frame, bg=CARD_BG, highlightthickness=1, highlightbackground=CARD_BORDER, bd=0)
            tk.Frame(card, bg=accent, height=2).pack(fill="x")
            inner = tk.Frame(card, bg=CARD_BG, padx=12, pady=8)
            inner.pack(fill="both", expand=True)
            tk.Label(inner, text=title.upper(), bg=CARD_BG, fg=TEXT_SOFT, font=("Arial", 8, "bold")).pack(anchor="w")
            value = tk.Label(inner, text="--", bg=CARD_BG, fg=TEXT_MAIN, font=("Consolas", 13, "bold"))
            value.pack(anchor="w", pady=(5, 1))
            tk.Label(inner, text=subtitle, bg=CARD_BG, fg=TEXT_MUTED, font=("Arial", 7)).pack(anchor="w")
            self.cards[key] = value
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
        for idx, card in enumerate(self.card_widgets):
            row = idx // columns
            col = idx % columns
            top_pad = 0 if row == 0 else 8
            card.grid(row=row, column=col, sticky="nsew", padx=5, pady=(top_pad, 0))

    def update(self, serial, rssi, noise, epoch, latency, latency_ema, raw_kdr, corr_kdr):
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


class GSNDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
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
        self.default_hosts = {}

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
        style.configure("Title.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=("Arial", 12, "bold"))
        style.configure("MetricTitle.TLabel", background=CARD_BG, foreground=TEXT_SOFT, font=("Arial", 9, "bold"))
        style.configure("Muted.TLabel", background=CARD_BG, foreground=TEXT_MUTED, font=("Arial", 10))
        style.configure("Header.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=("Arial", 16, "bold"))
        style.configure("SubHeader.TLabel", background=CARD_BG, foreground=TEXT_MUTED, font=("Arial", 9))
        style.configure("StatusBanner.TLabel", background=CARD_BG, foreground=ACCENT_BLUE, font=("Consolas", 10, "bold"))
        style.configure("SectionTitle.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=("Arial", 11, "bold"))
        style.configure("SectionMeta.TLabel", background=CARD_BG, foreground=TEXT_SOFT, font=("Arial", 8))
        style.configure("Badge.TLabel", background=CARD_BG_ALT, foreground=TEXT_MUTED, font=("Arial", 8, "bold"))
        style.configure("TButton", font=("Arial", 10, "bold"), padding=(9, 6))
        style.configure("TMenubutton", font=("Arial", 10, "bold"), padding=(7, 5))
        style.configure("StatusIdle.TLabel", background=CARD_BG, foreground=TEXT_MUTED, font=("Arial", 10, "bold"))
        style.configure("StatusWarn.TLabel", background=CARD_BG, foreground=ACCENT_AMBER, font=("Arial", 10, "bold"))
        style.configure("StatusGood.TLabel", background=CARD_BG, foreground=ACCENT_GREEN, font=("Arial", 10, "bold"))
        style.configure("StatusBad.TLabel", background=CARD_BG, foreground="#fb7185", font=("Arial", 10, "bold"))
        style.configure(
            "Panel.TCombobox",
            fieldbackground=CARD_BG_ALT,
            background=CARD_BG_ALT,
            foreground=TEXT_MAIN,
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
            font=("Arial", 9),
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
        tk.Label(status_shell, text="BACKEND", bg=CARD_BG_ALT, fg=TEXT_SOFT, font=("Arial", 8, "bold")).pack(anchor="e")
        self.status_banner = tk.Label(status_shell, text="OFF", bg=CARD_BG_ALT, fg=TEXT_MUTED, font=("Consolas", 9, "bold"))
        self.status_banner.pack(anchor="e", pady=(3, 0))

        controls = tk.Frame(top_inner, bg=CARD_BG)
        controls.pack(side="right")
        self.start_btn = ttk.Button(controls, text="Start", command=self.start_backend)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.pause_btn = ttk.Button(controls, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=(0, 6))
        self.clear_btn = ttk.Button(controls, text="Clear", command=self.clear_log)
        self.clear_btn.pack(side="left", padx=(0, 6))
        self.reset_layout_btn = ttk.Button(controls, text="Reset", command=self.reset_layout)
        self.reset_layout_btn.pack(side="left", padx=(0, 9))
        tk.Frame(controls, bg=CARD_BORDER_SOFT, width=1, height=26).pack(side="left", padx=(0, 9), pady=2)
        self.panel_menu_button = ttk.Menubutton(controls, text="Modules")
        self.panel_menu_button.pack(side="left", padx=(0, 6))
        self.layout_menu_button = ttk.Menubutton(controls, text="Presets")
        self.layout_menu_button.pack(side="left")

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
        self.panel_hosts["auxiliary"] = PanelHost(left_stack, "auxiliary", "Empty slot")
        self.panel_hosts["secondary"] = PanelHost(right_stack, "secondary", "Empty slot")
        self.panel_hosts["tertiary"] = PanelHost(right_stack, "tertiary", "Empty slot")
        self.panel_hosts["analytics_left"] = PanelHost(bottom_row, "analytics_left", "Empty slot")
        self.panel_hosts["analytics_right"] = PanelHost(bottom_row, "analytics_right", "Empty slot")
        self.panel_hosts["footer"] = PanelHost(footer, "footer", "Empty slot")

        left_stack.add(self.panel_hosts["primary"], weight=6)
        left_stack.add(self.panel_hosts["auxiliary"], weight=1)
        right_stack.add(self.panel_hosts["secondary"], weight=1)
        right_stack.add(self.panel_hosts["tertiary"], weight=1)
        main_row.add(left_stack, weight=6)
        main_row.add(right_stack, weight=2)
        bottom_row.add(self.panel_hosts["analytics_left"], weight=1)
        bottom_row.add(self.panel_hosts["analytics_right"], weight=1)
        self.panel_hosts["footer"].pack(fill="both", expand=True)

        self.default_hosts = {
            "media_main": self.panel_hosts["primary"],
            "control_panel": self.panel_hosts["auxiliary"],
            "text_main": self.panel_hosts["analytics_left"],
            "text_aux": self.panel_hosts["footer"],
            "chart_main": self.panel_hosts["secondary"],
            "chart_aux": self.panel_hosts["tertiary"],
            "text_log": self.panel_hosts["analytics_right"],
        }
        self._init_panels()
        self._init_panel_menu()
        self._init_layout_menu()
        self.after(0, self.apply_auto_layout_preset)

    def _init_panels(self):
        self.panels = {
            "media_main": VideoModulePanel(self, "media_main", "Decrypted Video", "video"),
            "control_panel": ControlModulePanel(self, "control_panel", "System State", "session"),
            "text_main": TextModulePanel(self, "text_main", "Telemetry Snapshot", "log"),
            "text_aux": TextModulePanel(self, "text_aux", "Epoch History", "epoch_history"),
            "chart_main": ChartModulePanel(self, "chart_main", "Latency Trend", "latency"),
            "chart_aux": ChartModulePanel(self, "chart_aux", "RSSI / Noise", "signal"),
            "text_log": TextModulePanel(self, "text_log", "System Log", "epoch_history"),
        }
        self.default_visible_panels = {
            "media_main": True,
            "control_panel": False,
            "text_main": True,
            "text_aux": False,
            "chart_main": True,
            "chart_aux": True,
            "text_log": True,
        }
        for key, panel in self.panels.items():
            panel.visible = self.default_visible_panels.get(key, True)
            if panel.visible:
                panel.show()

    def _init_panel_menu(self):
        menu = tk.Menu(self.panel_menu_button, tearoff=False)
        labels = {
            "media_main": "Decrypted Video",
            "control_panel": "System State",
            "text_main": "Telemetry Snapshot",
            "text_aux": "Epoch History",
            "chart_main": "Latency Trend",
            "chart_aux": "RSSI / Noise",
            "text_log": "System Log",
        }
        for key, label in labels.items():
            var = tk.BooleanVar(value=self.panels[key].visible)
            self.panel_visibility_vars[key] = var
            menu.add_checkbutton(label=label, variable=var, command=lambda k=key: self.toggle_panel(k))
        self.panel_menu_button["menu"] = menu

    def _init_layout_menu(self):
        menu = tk.Menu(self.layout_menu_button, tearoff=False)
        menu.add_command(label="Auto Detect", command=self.apply_auto_layout_preset)
        menu.add_command(label="Balanced", command=lambda: self.apply_layout_preset("balanced"))
        menu.add_command(label="Wide Video", command=lambda: self.apply_layout_preset("wide_video"))
        menu.add_command(label="Focus Video", command=lambda: self.apply_layout_preset("focus_video"))
        menu.add_command(label="Analysis", command=lambda: self.apply_layout_preset("analysis"))
        self.layout_menu_button["menu"] = menu

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
        rejected_epochs = set()
        while True:
            try:
                data, addr = sock.recvfrom(1024*1024)
                parts = data.decode(errors="ignore").split()
                if len(parts) != 5 or parts[0] != "R":
                    continue
                epoch = int(parts[1])
                serial = int(parts[2])
                helper = parts[3]
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
                    with self.state_obj.lock:
                        self.state_obj.video_status = (
                            f"Waiting for local CSI serial={serial} before BCH correction."
                        )
                        self.state_obj.video_status_level = "warn"
                    self.log(f"Waiting for local CSI serial={serial} for epoch={epoch}.")
                    continue

                try:
                    corrected = bch_decode_key(local_raw, helper)
                except ValueError as e:
                    with self.state_obj.lock:
                        self.state_obj.video_status = (
                            f"BCH correction failed for epoch={epoch}. Waiting for next helper."
                        )
                        self.state_obj.video_status_level = "warn"
                    self.log(f"BCH correction failed for epoch={epoch}, serial={serial}: {e}")
                    continue
                aes = sha256.sha_byte(corrected)
                if not verify_key_confirm(aes, epoch, serial, helper, confirm):
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
                self.log(f"BCH loop error: {e}")
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
                noise_hist = list(self.state_obj.noise_hist)
                running = self.state_obj.started
                model_loaded = self.state_obj.model_loaded
                serial_ok = self.state_obj.serial_connected
                rx_ok = self.state_obj.receiver_started
                bch_ok = self.state_obj.bch_started

            self.stats_strip.update(serial, rssi, noise, epoch, latency, latency_ema, raw_kdr, corr_kdr)
            snapshot = {
                "serial": serial,
                "rssi": rssi,
                "noise": noise,
                "epoch": epoch,
                "latency": latency,
                "latency_ema": latency_ema,
                "frame_time": frame_time,
                "raw_kdr": raw_kdr,
                "corr_kdr": corr_kdr,
                "frame": frame,
                "gsn_raw": gsn_raw,
                "aes_key": aes_key,
                "video_status": video_status,
                "video_status_level": video_status_level,
                "keys_by_epoch": keys_by_epoch,
                "kdr_raw_hist": kdr_raw_hist,
                "kdr_corr_hist": kdr_corr_hist,
                "lat_hist": lat_hist,
                "lat_ema_hist": lat_ema_hist,
                "rssi_hist": rssi_hist,
                "noise_hist": noise_hist,
                "started": running,
                "model_loaded": model_loaded,
                "serial_ok": serial_ok,
                "rx_ok": rx_ok,
                "bch_ok": bch_ok,
            }
            for panel in self.panels.values():
                panel.update_snapshot(snapshot)
            if running:
                ready_count = sum(1 for item in (model_loaded, serial_ok, rx_ok, bch_ok) if item)
                self.status_banner.config(
                    text=f"ON {ready_count}/4",
                    fg=ACCENT_GREEN if ready_count == 4 else ACCENT_AMBER,
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
