# face_gui_hue.py  (Python 3.7–3.9 compatible)
import os
import cv2
import glob
import time
import json
import queue
import threading
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

# Optional: MTCNN for face detection/alignment
try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except Exception:
    HAS_MTCNN = False

import requests
from typing import Optional, Dict, List, Tuple
# ---- Logging safety for X11 ----
DISABLE_EMOJI = True          # set True on X11/SSH; False if you really want emojis locally
MAX_LOG_CHARS = 20000         # keep Text buffer small to avoid X glyph cache issues

def _sanitize_log_text(s: str) -> str:
    if DISABLE_EMOJI:
        # Strip non-ASCII (removes emoji and other complex glyphs)
        s = s.encode("ascii", "ignore").decode("ascii")
    # Hard cap log size (safety — also enforced in logln)
    if len(s) > MAX_LOG_CHARS:
        s = s[-MAX_LOG_CHARS:]
    return s


# ---------------------------- Config & Paths ----------------------------
SIGNATURE_DIR = "./signatures"
DATASET_DIR   = "./dataset"     # cropped images per person (audit)
os.makedirs(SIGNATURE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Enroll timing & UI appearance ---
CAPTURE_COUNTDOWN = 3      # seconds shown before each photo (3..2..1)
CAPTURE_PAUSE_AFTER = 1.0  # short pause after each capture (seconds)
PROMPT_FONT = ("Arial", 18, "bold")  # large prompt label font
BANNER_FONT_SCALE = 1.0    # overlay text scale on video
BANNER_THICKNESS = 2       # overlay text thickness


DEFAULT_THRESHOLD = 0.40
DEFAULT_TIMEOUT   = 10
CAMERA_INDEX      = 0
ENROLL_IMAGES     = 8

PROMPTS = [
    "Look straight / Pogled ravno",
    "Turn slightly left / Malo ulijevo",
    "Turn slightly right / Malo udesno",
    "Look slightly up / Malo prema gore",
    "Look slightly down / Malo prema dolje",
    "Smile / Nasmijesite se",
    "Serious / Ozbiljno",
    "Relax / Opusteno",
]

# ---------------------------- ML Models ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

if HAS_MTCNN:
    mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)
else:
    mtcnn = None  # fallback path will center-crop

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)

def pil_to_embedding(pil_img: Image.Image) -> np.ndarray:
    t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(t).squeeze().cpu().numpy().astype(np.float32)
    return l2_normalize(emb)

def detect_align(pil_img: Image.Image) -> Optional[Image.Image]:
    """Return aligned 160x160 PIL face or None if not found."""
    if mtcnn is not None:
        face_t = mtcnn(pil_img)
        if face_t is None:
            return None
        face_np = (face_t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(face_np)
    # fallback: simple center crop to square + resize
    w, h = pil_img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    crop = pil_img.crop((left, top, left + side, top + side)).resize((160, 160))
    return crop

# ---------------------------- Hue Helpers ----------------------------
def hue_put(bridge_ip: str, username: str, path: str, payload: dict,
            tries: int = 2, timeout: int = 3) -> Tuple[bool, str]:
    url = "http://{}/api/{}{}".format(bridge_ip, username, path)
    for i in range(tries):
        try:
            r = requests.put(url, json=payload, timeout=timeout)
            ok = r.status_code == 200
            return ok, r.text
        except Exception as e:
            if i == tries - 1:
                return False, str(e)
            time.sleep(0.2)

def hue_green(bridge_ip: str, username: str, light_id: str) -> Tuple[bool, str]:
    return hue_put(bridge_ip, username, "/lights/{}/state".format(light_id),
                   {"on": True, "effect": "none", "alert": "none",
                    "hue": 25500, "sat": 254, "bri": 254})

def hue_red(bridge_ip: str, username: str, light_id: str) -> Tuple[bool, str]:
    return hue_put(bridge_ip, username, "/lights/{}/state".format(light_id),
                   {"on": True, "effect": "none", "alert": "none",
                    "hue": 0, "sat": 254, "bri": 254})

def hue_off(bridge_ip: str, username: str, light_id: str, transition_sec: float = 1.5) -> bool:
    tt = max(1, int(transition_sec * 10))  # 1/10 s units
    ok1, _ = hue_put(bridge_ip, username, "/lights/{}/state".format(light_id),
                     {"effect": "none", "alert": "none", "bri": 1, "transitiontime": tt})
    ok2, _ = hue_put(bridge_ip, username, "/lights/{}/state".format(light_id),
                     {"on": False, "transitiontime": tt})
    return bool(ok1 and ok2)

# ---------------------------- Signature I/O ----------------------------
def load_signatures() -> Dict[str, np.ndarray]:
    db = {}
    for p in glob.glob(os.path.join(SIGNATURE_DIR, "*.npy")):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            vec = np.load(p).astype(np.float32)
            db[name] = l2_normalize(vec)
        except Exception:
            pass
    return db

def save_signature(name: str, emb_list: List[np.ndarray]) -> str:
    sig = l2_normalize(np.mean(emb_list, axis=0).astype(np.float32))
    out = os.path.join(SIGNATURE_DIR, "{}.npy".format(name))
    np.save(out, sig)
    return out

def recognize_best(emb: np.ndarray, db: Dict[str, np.ndarray]) -> Tuple[str, float]:
    best_name, best_sim = "", float("-inf")
    for n, v in db.items():
        sim = float(np.dot(v, emb))
        if sim > best_sim:
            best_sim, best_name = sim, n
    return best_name, best_sim

# ---------------------------- Camera Thread ----------------------------
class Camera:
    def __init__(self, index: int = 0):
        self.cap = None
        self.index = index
        self.running = False
        self.frame_lock = threading.Lock()
        self.frame = None

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera.")
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.frame_lock:
                    self.frame = f.copy()
            time.sleep(0.01)

    def get_frame(self):
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

# ---------------------------- GUI App ----------------------------
class FaceGUI(tk.Tk):
    """Tkinter GUI for enrollment + recognition + Hue control, dark themed, larger UI with status bar."""

    # Keep GUI alive on any Tk callback exception
    def report_callback_exception(self, exc, val, tb):
        import traceback
        msg = "".join(traceback.format_exception(exc, val, tb))
        self.logln("[!] Unhandled error (kept app running):")
        self.logln(msg)

    def __init__(self):
        super(FaceGUI, self).__init__()
        self.title("Face Demo + Hue")
        self.geometry("1280x820")
        self.configure(padx=10, pady=10)

        # ---- Styles / Theme ----
        self._init_style()

        # Camera + DB
        self.cam = Camera(CAMERA_INDEX)
        self.db = load_signatures()

        # ---- Main layout: paned center + status bar at bottom ----
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(paned, style="Card.TFrame", padding=14)
        right = ttk.Frame(paned, padding=4, style="TFrame")
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        # -------- Left: Preview card --------
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        header = ttk.Label(left, text="Camera Preview", style="Header.TLabel")
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        preview_wrap = ttk.Frame(left, style="Inset.TFrame", padding=6)
        preview_wrap.grid(row=1, column=0, sticky="nsew")

        self.preview = tk.Label(preview_wrap, bg="#05070c")
        self.preview.grid(row=0, column=0, sticky="nsew")
        preview_wrap.rowconfigure(0, weight=1)
        preview_wrap.columnconfigure(0, weight=1)

        # Result label (big status text)
        self.result_label = tk.Label(
            left, text="", font=("Arial", 28, "bold"),
            fg="#22c55e", bg=self._bg_body
        )
        self.result_label.grid(row=2, column=0, sticky="we", pady=(14, 0))

        # ---- Right: stacked cards ----
        right.grid_columnconfigure(0, weight=1)

        # ACTIONS card
        actions = ttk.Frame(right, style="Card.TFrame", padding=16)
        actions.grid(row=0, column=0, sticky="we", pady=(0, 2))
        actions.grid_columnconfigure(0, weight=1)

        ttk.Label(actions, text="Actions", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Candidate name row (label + entry)
        name_row = ttk.Frame(actions, style="Card.TFrame")
        name_row.grid(row=1, column=0, sticky="we", pady=(0, 8))
        ttk.Label(name_row, text="Candidate name:", style="Subtle.TLabel").pack(side="left")
        self.name_var = tk.StringVar()
        ttk.Entry(
            name_row, textvariable=self.name_var, width=20,
            font=("Arial", 14), style="Input.TEntry"
        ).pack(side="left", padx=(10, 0))

        # Big buttons row
        bigbtn_row = ttk.Frame(actions, style="Card.TFrame")
        bigbtn_row.grid(row=2, column=0, sticky="we", pady=(4, 0))
        ttk.Button(bigbtn_row, text="Start Camera", style="Accent.TButton",
                   command=self.start_camera).pack(side="left", padx=5, pady=3)
        ttk.Button(bigbtn_row, text="Stop Camera", style="Accent.TButton",
                   command=self.stop_camera).pack(side="left", padx=5, pady=3)
        ttk.Button(bigbtn_row, text="Enroll", style="Primary.TButton",
                   command=self.enroll).pack(side="left", padx=10, pady=3)
        ttk.Button(bigbtn_row, text="Recognize", style="Primary.TButton",
                   command=self.recognize_once).pack(side="left", padx=5, pady=3)

        # Big prompt label (visible instructions)
        self.prompt_label = ttk.Label(
            actions, text="", font=("Arial", 20, "bold"),
            style="Prompt.TLabel", wraplength=380
        )
        self.prompt_label.grid(row=3, column=0, sticky="we", pady=(10, 0))

        # DATABASE card
        db_card = ttk.Frame(right, style="Card.TFrame", padding=6)
        db_card.grid(row=1, column=0, sticky="we", pady=(0, 4))
        ttk.Label(db_card, text="Database", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        # Listbox + scrollbar
        lb_wrap = ttk.Frame(db_card, style="Card.TFrame")
        lb_wrap.grid(row=1, column=0, sticky="we")
        lb_wrap.grid_columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(
            lb_wrap, height=4, relief="flat",
            highlightthickness=1, highlightcolor="#4b5563",
            bg="#020617", fg="#e5e7eb",
            selectbackground="#1f2937", selectforeground="#e5e7eb",
            font=("Arial", 14)
        )
        self.listbox.grid(row=0, column=0, sticky="we")
        lb_scroll = ttk.Scrollbar(lb_wrap, orient="vertical", command=self.listbox.yview)
        lb_scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=lb_scroll.set)

        # DB buttons
        db_btns = ttk.Frame(db_card, style="Card.TFrame")
        db_btns.grid(row=2, column=0, sticky="we", pady=(8, 0))
        ttk.Button(db_btns, text="Refresh DB", style="TButton",
                   command=self.refresh_listbox).pack(side="left", padx=(0, 8))

        # SETTINGS card
        settings = ttk.Frame(right, style="Card.TFrame", padding=16)
        settings.grid(row=2, column=0, sticky="we", pady=(0, 2))
        settings.grid_columnconfigure(1, weight=1)
        ttk.Label(settings, text="Settings", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        ttk.Label(settings, text="Threshold:", style="Subtle.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        self.thresh_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        ttk.Scale(
            settings, from_=0.10, to=0.99, orient="horizontal",
            variable=self.thresh_var, style="Wide.Horizontal.TScale"
        ).grid(row=1, column=1, sticky="we", pady=2)
        self.thresh_label = ttk.Label(settings, text=f"{DEFAULT_THRESHOLD:.2f}", style="Value.TLabel")
        self.thresh_label.grid(row=1, column=2, sticky="e", padx=(10, 0))

        ttk.Label(settings, text="Light timeout (s):", style="Subtle.TLabel").grid(row=2, column=0, sticky="w", pady=4)
        self.timeout_var = tk.IntVar(value=DEFAULT_TIMEOUT)
        ttk.Spinbox(
            settings, from_=3, to=120, textvariable=self.timeout_var,
            width=6, style="Input.TSpinbox", font=("Arial", 14)
        ).grid(row=2, column=1, sticky="w")

        ttk.Separator(settings, orient="horizontal").grid(row=3, column=0, columnspan=3, sticky="we", pady=10)

        ttk.Label(settings, text="Hue Bridge IP:", style="Subtle.TLabel").grid(row=4, column=0, sticky="w", pady=4)
        self.hue_ip = tk.StringVar(value="192.168.1.100")
        ttk.Entry(settings, textvariable=self.hue_ip,
                  style="Input.TEntry", font=("Arial", 14)).grid(
            row=4, column=1, columnspan=2, sticky="we", pady=1
        )

        ttk.Label(settings, text="Hue Username:", style="Subtle.TLabel").grid(row=5, column=0, sticky="w", pady=4)
        self.hue_user = tk.StringVar(value="v84V00gmP09XseQaCEnqkIKT68yUXm8uLD8s9LJi")
        ttk.Entry(settings, textvariable=self.hue_user,
                  style="Input.TEntry", font=("Arial", 14)).grid(
            row=5, column=1, columnspan=2, sticky="we", pady=1
        )

        ttk.Label(settings, text="Hue Light ID:", style="Subtle.TLabel").grid(row=6, column=0, sticky="w", pady=4)
        self.hue_light = tk.StringVar(value="4")
        ttk.Entry(settings, textvariable=self.hue_light, width=10,
                  style="Input.TEntry", font=("Arial", 14)).grid(
            row=6, column=1, sticky="w", pady=1
        )

        # LOG card
        log_card = ttk.Frame(right, style="Card.TFrame", padding=10)
        log_card.grid(row=3, column=0, sticky="nsew", pady=0)
        right.grid_rowconfigure(3, weight=1)
        ttk.Label(log_card, text="Log", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))

        log_wrap = ttk.Frame(log_card, style="Card.TFrame")
        log_wrap.grid(row=1, column=0, sticky="nsew")
        log_card.grid_rowconfigure(1, weight=1)
        log_wrap.grid_columnconfigure(0, weight=1)

        self.log = tk.Text(
            log_wrap, height=10, wrap="word", relief="flat",
            highlightthickness=1, highlightbackground="#4b5563",
            bg="#020617", fg="#e5e7eb", insertbackground="#e5e7eb",
            font=("Consolas", 14)
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log.config(yscrollcommand=log_scroll.set)

        # ---- Status bar at the bottom ----
        self.status_var = tk.StringVar()
        status_frame = ttk.Frame(self, style="Status.TFrame", padding=(10, 4))
        status_frame.grid(row=1, column=0, sticky="we", pady=(8, 0))
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side="left", fill="x")

        # init status
        self._cam_state = "STOPPED"
        self._hue_state = "IDLE"
        self._last_match = "—"
        self._set_status()

        # Only now it's safe to populate list and log
        self.refresh_listbox()

        # timers / banners
        self.off_timer = None
        self._banner_text = ""
        self._banner_expire = 0.0

        # periodic UI updates
        self.after(100, self.update_preview)
        self.after(100, self.update_threshold_label)

        # clear result text
        self.result_label.config(text="")

    # ---------- Styling ----------
    def _init_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Dark palette
        self._bg_body = "#020617"   # almost black
        self._bg_card = "#030712"   # dark card
        self._bg_inset = "#0b1220"  # slightly raised area
        self._fg_text = "#e5e7eb"   # light gray
        self._fg_muted = "#9ca3af"  # muted gray
        self._accent = "#3b82f6"    # blue
        self._primary = "#22c55e"   # green

        self.configure(bg=self._bg_body)

        style.configure(".", background=self._bg_body, foreground=self._fg_text)
        style.configure("TFrame", background=self._bg_body)
        style.configure("Card.TFrame", background=self._bg_card, relief="flat")
        style.configure("Inset.TFrame", background=self._bg_inset, relief="flat")

        style.configure("Header.TLabel", background=self._bg_card,
                        foreground=self._fg_text, font=("Arial", 16, "bold"))
        style.configure("Subtle.TLabel", background=self._bg_card,
                        foreground=self._fg_muted, font=("Arial", 13))
        style.configure("Value.TLabel", background=self._bg_card,
                        foreground=self._fg_text, font=("Consolas", 14, "bold"))
        style.configure("Prompt.TLabel", background=self._bg_card,
                        foreground=self._fg_text)

        # Status bar styles
        style.configure("Status.TFrame", background="#020617")
        style.configure("Status.TLabel", background="#020617",
                        foreground="#9ca3af", font=("Consolas", 12))

        # Input text color (#030712) in entries/spinboxes (bigger)
        style.configure(
            "Input.TEntry",
            foreground="#030712",
            fieldbackground="#ffffff",
            insertcolor="#030712",
            padding=(6, 4)
        )
        style.configure(
            "Input.TSpinbox",
            foreground="#030712",
            fieldbackground="#ffffff",
            arrowsize=14,
            padding=(4, 4)
        )

        # Base button (bigger)
        style.configure("TButton",
                        padding=(14, 8),
                        font=("Arial", 13),
                        background="#111827",
                        foreground=self._fg_text)
        style.map("TButton",
                  background=[("active", "#1f2937")])

        # Accent button (blue) – bigger
        style.configure("Accent.TButton",
                        padding=(16, 10),
                        font=("Arial", 14, "bold"),
                        background=self._accent,
                        foreground="#0b1120")
        style.map("Accent.TButton",
                  background=[("active", "#2563eb")],
                  foreground=[("!disabled", "#0b1120")])

        # Primary button (green) – bigger
        style.configure("Primary.TButton",
                        padding=(16, 10),
                        font=("Arial", 14, "bold"),
                        background=self._primary,
                        foreground="#022c22")
        style.map("Primary.TButton",
                  background=[("active", "#16a34a")],
                  foreground=[("!disabled", "#022c22")])

        # Wider / prettier horizontal scale for threshold
        style.configure(
            "Wide.Horizontal.TScale",
            troughcolor="#020617",
            background=self._bg_card,
            thickness=12,
            sliderlength=26,
            borderwidth=0
        )

    # ---------------- Status helper ----------------
    def _set_status(self, cam=None, hue=None, last=None):
        if cam is not None:
            self._cam_state = cam
        if hue is not None:
            self._hue_state = hue
        if last is not None:
            self._last_match = last

        txt = f"Camera: {self._cam_state}   |   Hue: {self._hue_state}   |   Last match: {self._last_match}"
        self.status_var.set(txt)

    # ---------------- UI helpers ----------------
    def logln(self, msg: str):
        msg = _sanitize_log_text(msg)
        try:
            content = self.log.get("1.0", "end-1c")
            if len(content) > MAX_LOG_CHARS:
                cut_at = int(len(content) * 0.5)
                self.log.delete("1.0", f"1.0+{cut_at}c")
        except Exception:
            pass
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def refresh_listbox(self):
        """Re-scan signatures/ on disk, rebuild DB, and refresh the listbox."""
        self.db = load_signatures()
        names = []
        for p in glob.glob(os.path.join(SIGNATURE_DIR, "*.npy")):
            if os.path.isfile(p):
                names.append(os.path.splitext(os.path.basename(p))[0])
        names = sorted(names)

        self.listbox.delete(0, "end")
        for name in names:
            self.listbox.insert("end", name)

        if hasattr(self, "log"):
            self.logln(f"[DB] Loaded {len(names)} signature(s) from disk.")

    def update_threshold_label(self):
        self.thresh_label.config(text=f"{self.thresh_var.get():.2f}")
        self.after(200, self.update_threshold_label)

    def _set_banner(self, text: str, seconds: float):
        self._banner_text = text
        self._banner_expire = time.time() + seconds

    def _clear_banner(self):
        self._banner_text = ""
        self._banner_expire = 0.0

    def start_camera(self):
        try:
            if self.cam.running:
                self.logln("Camera already running.")
                self._set_status(cam="RUNNING")
                return
            self.cam.cap = cv2.VideoCapture(self.cam.index)
            if not self.cam.cap.isOpened():
                messagebox.showerror("Camera", "Cannot open camera.")
                self.logln("Cannot open camera.")
                self._set_status(cam="ERROR")
                return
            self.cam.running = True
            threading.Thread(target=self.cam._loop, daemon=True).start()
            self.logln("Camera started.")
            self._set_status(cam="RUNNING")
        except Exception as e:
            self.logln(f"Camera start error: {e}")
            self._set_status(cam="ERROR")

    def stop_camera(self):
        self.cam.stop()
        self.preview.config(image="")
        self.logln("Camera stopped.")
        self._set_status(cam="STOPPED")

    def update_preview(self):
        frame = self.cam.get_frame()
        if frame is not None:
            # Banner overlay if active
            if self._banner_text and time.time() < self._banner_expire:
                overlay = frame.copy()
                h, w, _ = frame.shape
                bar_h = int(0.12 * h)
                cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
                alpha = 0.55
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                lines = self._banner_text.split("\n")
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (12, 30 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                                BANNER_FONT_SCALE, (255, 255, 255), BANNER_THICKNESS, cv2.LINE_AA)

            # BGR -> RGB, fit to label and cap width
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            max_w = min(960, int(self.winfo_width() * 0.62) if self.winfo_width() > 0 else w)
            scale = min(float(max_w) / float(w), 1.0)
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            im = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=im)
            self.preview.imgtk = imgtk
            self.preview.configure(image=imgtk)
        self.after(30, self.update_preview)

    # ---------------- Enroll (slow + visible + crash-proof) ----------------
    def enroll(self):
        try:
            name = self.name_var.get().strip()
            if not name:
                messagebox.showwarning("Enroll", "Enter candidate name first.")
                return
            if self.cam.get_frame() is None:
                messagebox.showwarning("Enroll", "Start the camera first.")
                return

            person_dir = os.path.join(DATASET_DIR, name)
            os.makedirs(person_dir, exist_ok=True)

            self.logln(f"Enrolling: {name}")
            embs = []
            captured = 0

            for i in range(ENROLL_IMAGES):
                prompt = PROMPTS[i] if i < len(PROMPTS) else "Pose"
                self.prompt_label.config(text=f"{prompt}  [{i + 1} of {ENROLL_IMAGES}]")
                # Countdown on preview
                for tsec in range(CAPTURE_COUNTDOWN, 0, -1):
                    self._set_banner(f"{prompt}\nCapturing in {tsec}...", seconds=1.0)
                    self.update()
                    time.sleep(1.0)

                # Try several frames to find a face
                face_pil = None
                for _ in range(12):
                    frame = self.cam.get_frame()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    face_pil = detect_align(pil)
                    if face_pil is not None:
                        break
                    time.sleep(0.1)

                if face_pil is None:
                    self.logln("  [!] No face detected, skipping this sample.")
                    self._clear_banner()
                    continue

                face_path = os.path.join(person_dir, f"face_{i + 1}.jpg")
                try:
                    face_pil.save(face_path, quality=95)
                except Exception as e:
                    self.logln(f"  [!] Could not save {face_path}: {e}")
                    self._clear_banner()
                    continue

                try:
                    emb = pil_to_embedding(face_pil)
                    embs.append(emb)
                    captured += 1
                    self.logln(f"  Saved {face_path}")
                except Exception as e:
                    self.logln(f"  [!] Embedding failed for sample {i + 1}: {e}")

                self.update()
                time.sleep(CAPTURE_PAUSE_AFTER)
                self._clear_banner()

            self.prompt_label.config(text="")

            if captured == 0:
                self.logln("[X] Enrollment failed: no valid face samples.")
                return

            try:
                out_path = save_signature(name, embs)
            except Exception as e:
                self.logln(f"[X] Could not save signature: {e}")
                return

            self.db = load_signatures()
            self.refresh_listbox()
            self.logln(f"[OK] Signature saved: {out_path} (samples: {captured})")

        except Exception as e:
            self.logln(f"[!] Error during enrollment: {e}")

    # ---------------- Recognize (crash-proof) ----------------
    def recognize_once(self):
        try:
            if self.cam.get_frame() is None:
                messagebox.showwarning("Recognize", "Start the camera first.")
                return
            if not self.db:
                self.db = load_signatures()
            if not self.db:
                messagebox.showwarning("Recognize", "No signatures found. Enroll first.")
                return

            self._set_banner("Recognition: look at the camera", seconds=2.0)

            frame = self.cam.get_frame()
            if frame is None:
                self.logln("No frame from camera.")
                self._clear_banner()
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            face_pil = detect_align(pil)
            if face_pil is None:
                self.logln("[!] No face detected.")
                self._clear_banner()
                return

            emb = pil_to_embedding(face_pil)
            name, sim = recognize_best(emb, self.db)
            thr = float(self.thresh_var.get())
            recognized = sim >= thr

            bridge_ip = self.hue_ip.get().strip()
            user = self.hue_user.get().strip()
            light_id = self.hue_light.get().strip()
            timeout_s = int(self.timeout_var.get())

            if recognized:
                self.result_label.config(text=f"Recognized: {name}", fg="#22c55e")
                self.logln(f"[OK] Recognized: {name} (sim={sim:.3f} >= {thr:.2f}) -> GREEN")
                self._set_status(last=f"{name} ({sim:.2f})")
                #ok, resp = hue_green(bridge_ip, user, light_id)
                #if ok:
                    #self._set_status(hue="ON")
                #else:
                    #self._set_status(hue="ERROR")
                    #self.logln(f"[Hue] Error: {resp}")
            else:
                close = name if name else "-"
                self.result_label.config(text="Unknown", fg="#f97373")
                self.logln(f"[X] Unknown (closest: {close}, sim={sim:.3f} < {thr:.2f}) -> RED")
                self._set_status(last=f"Unknown ({sim:.2f})")
                #ok, resp = hue_red(bridge_ip, user, light_id)
                #if ok:
                    #self._set_status(hue="ON")
                #else:
                    #self._set_status(hue="ERROR")
                    #self.logln(f"[Hue] Error: {resp}")

            if self.off_timer and self.off_timer.is_alive():
                pass
            #self.logln(f"[LIGHT] Turning off in {timeout_s}s...")
            #self.off_timer = threading.Timer(timeout_s, self._turn_off_hue_safe,
                                             #args=(bridge_ip, user, light_id))
            #self.off_timer.daemon = True
            #self.off_timer.start()

            self._clear_banner()
            self.after(5000, lambda: self.result_label.config(text=""))
            self.logln("Ready for next recognition.")
        except Exception as e:
            self.logln(f"[!] Error during recognition: {e}")
            self._set_status(hue="ERROR")

    # Timer callback: turn Hue off safely
    def _turn_off_hue_safe(self, ip: str, user: str, light_id: str):
        try:
            ok = hue_off(ip, user, light_id, transition_sec=1.5)
            if ok:
                self.logln("[Hue] Off.")
                self._set_status(hue="OFF")
            else:
                self.logln("[Hue] Off may have failed.")
                self._set_status(hue="ERROR")
        except Exception as e:
            self.logln(f"[Hue] Off error: {e}")
            self._set_status(hue="ERROR")

# ---------------------------- Run ----------------------------
if __name__ == "__main__":
    app = FaceGUI()
    try:
        app.mainloop()
    finally:
        app.cam.stop()

