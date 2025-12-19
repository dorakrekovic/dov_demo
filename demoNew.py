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
    

    # Keep GUI alive on any Tk callback exception
    def report_callback_exception(self, exc, val, tb):
        import traceback
        msg = "".join(traceback.format_exception(exc, val, tb))
        self.logln("[!] Unhandled error (kept app running):")
        self.logln(msg)

    def __init__(self):
        super(FaceGUI, self).__init__()
        self.title("Face Demo + Hue")
        self.geometry("1100x740")
        self.configure(padx=10, pady=10)

        # Camera + DB
        self.cam = Camera(CAMERA_INDEX)
        self.db = load_signatures()

        # --- Left: Video preview ---
        self.preview = tk.Label(self, bg="#111")
        self.preview.grid(row=0, column=0, rowspan=30, sticky="nsew", padx=(0, 10))
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Big recognition result label under preview
        self.result_label = tk.Label(self, text="", font=("Arial", 24, "bold"), fg="#00aa00", bg="#111")
        self.result_label.grid(row=30, column=0, sticky="we", pady=(10, 0))

        # --- Right: Controls ---
        r = 0
        ttk.Label(self, text="Candidate name").grid(row=r, column=1, sticky="w"); r += 1
        self.name_var = tk.StringVar()
        # smaller entry + readable font
        ttk.Entry(self, textvariable=self.name_var, width=18, font=("Arial", 11)).grid(
            row=r, column=1, sticky="w", padx=(0, 10)
        ); r += 1

        ttk.Label(self, text="Signatures in DB").grid(row=r, column=1, sticky="w"); r += 1
        self.listbox = tk.Listbox(self, height=6)
        self.listbox.grid(row=r, column=1, sticky="we"); r += 1

        ttk.Separator(self, orient="horizontal").grid(row=r, column=1, sticky="we", pady=6); r += 1

        ttk.Label(self, text="Threshold (cosine)").grid(row=r, column=1, sticky="w"); r += 1
        self.thresh_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        ttk.Scale(self, from_=0.10, to=0.99, orient="horizontal",
                  variable=self.thresh_var).grid(row=r, column=1, sticky="we"); r += 1
        self.thresh_label = ttk.Label(self, text="{:.2f}".format(DEFAULT_THRESHOLD))
        self.thresh_label.grid(row=r-1, column=1, sticky="e")

        ttk.Label(self, text="Light timeout (s)").grid(row=r, column=1, sticky="w"); r += 1
        self.timeout_var = tk.IntVar(value=DEFAULT_TIMEOUT)
        ttk.Spinbox(self, from_=3, to=120, textvariable=self.timeout_var,
                    width=6).grid(row=r, column=1, sticky="w"); r += 1

        ttk.Separator(self, orient="horizontal").grid(row=r, column=1, sticky="we", pady=6); r += 1

        ttk.Label(self, text="Hue Bridge IP").grid(row=r, column=1, sticky="w"); r += 1
        self.hue_ip = tk.StringVar(value="10.19.4.128")
        ttk.Entry(self, textvariable=self.hue_ip).grid(row=r, column=1, sticky="we"); r += 1

        ttk.Label(self, text="Hue Username (API key)").grid(row=r, column=1, sticky="w"); r += 1
        self.hue_user = tk.StringVar(value="v84V00gmP09XseQaCEnqkIKT68yUXm8uLD8s9LJi")
        ttk.Entry(self, textvariable=self.hue_user).grid(row=r, column=1, sticky="we"); r += 1

        ttk.Label(self, text="Hue Light ID").grid(row=r, column=1, sticky="w"); r += 1
        self.hue_light = tk.StringVar(value="4")
        ttk.Entry(self, textvariable=self.hue_light, width=6).grid(row=r, column=1, sticky="w"); r += 1

        ttk.Separator(self, orient="horizontal").grid(row=r, column=1, sticky="we", pady=6); r += 1

        # Bigger buttons (styled)
        bigbtn_style = ttk.Style()
        bigbtn_style.configure("Big.TButton", font=("Arial", 12, "bold"), padding=(12, 8))

        btn_row = tk.Frame(self)
        btn_row.grid(row=r, column=1, sticky="we"); r += 1
        ttk.Button(btn_row, text="▶ Start Camera", style="Big.TButton",
                   command=self.start_camera).pack(side="left", padx=4, pady=2)
        ttk.Button(btn_row, text="■ Stop Camera", style="Big.TButton",
                   command=self.stop_camera).pack(side="left", padx=4, pady=2)

        btn_row2 = tk.Frame(self)
        btn_row2.grid(row=r, column=1, sticky="we"); r += 1
        ttk.Button(btn_row2, text="Enroll", style="Big.TButton",
                   command=self.enroll).pack(side="left", padx=4, pady=2)
        ttk.Button(btn_row2, text="Recognize", style="Big.TButton",
                   command=self.recognize_once).pack(side="left", padx=4, pady=2)
        ttk.Button(btn_row2, text="Refresh DB",
                   command=self.refresh_listbox).pack(side="left", padx=4, pady=2)

        ttk.Separator(self, orient="horizontal").grid(row=r, column=1, sticky="we", pady=6); r += 1

        ttk.Label(self, text="Log").grid(row=r, column=1, sticky="w"); r += 1
        # Bigger, more readable log
        self.log = tk.Text(self, height=16, wrap="word", relief="solid", borderwidth=1)
        self.log.grid(row=r, column=1, sticky="nsew", pady=(4, 4)); r += 1
        self.log.configure(font=("Consolas", 11))
        self.columnconfigure(1, weight=0)

        # Only now it's safe to populate list and log
        self.refresh_listbox()

        # Big prompt text (visible instructions)
        self.prompt_label = ttk.Label(self, text="", font=PROMPT_FONT, foreground="#222", wraplength=340)
        self.prompt_label.grid(row=r, column=1, sticky="we", pady=(8, 0)); r += 1

        # periodic UI updates
        self.after(100, self.update_preview)
        self.after(100, self.update_threshold_label)

        self.off_timer = None

        # banner overlay state for preview
        self._banner_text = ""
        self._banner_expire = 0.0

        # clear result label
        self.result_label.config(text="")

    # ---------------- UI helpers ----------------
    def logln(self, msg: str):
        # sanitize and cap size for X11 stability
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
        self.thresh_label.config(text="{:.2f}".format(self.thresh_var.get()))
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
                return
            self.cam.cap = cv2.VideoCapture(self.cam.index)
            if not self.cam.cap.isOpened():
                messagebox.showerror("Camera", "Cannot open camera.")
                self.logln("Cannot open camera.")
                return
            self.cam.running = True
            threading.Thread(target=self.cam._loop, daemon=True).start()
            self.logln("Camera started.")
        except Exception as e:
            self.logln("Camera start error: {}".format(e))

    def stop_camera(self):
        self.cam.stop()
        self.preview.config(image="")
        self.logln("Camera stopped.")

    def update_preview(self):
        frame = self.cam.get_frame()
        if frame is not None:
            # Draw banner overlay if active
            if self._banner_text and time.time() < self._banner_expire:
                overlay = frame.copy()
                h, w, _ = frame.shape
                bar_h = int(0.12 * h)
                cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
                alpha = 0.55
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                # Place text (supports two lines: prompt + countdown)
                lines = self._banner_text.split("\n")
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (12, 28 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                                BANNER_FONT_SCALE, (255, 255, 255), BANNER_THICKNESS, cv2.LINE_AA)

            # BGR -> RGB, fit to label and cap width for X11 stability
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            max_w = min(900, int(self.winfo_width() * 0.62) if self.winfo_width() > 0 else w)
            scale = min(float(max_w) / float(w), 1.0)
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            im = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=im)
            self.preview.imgtk = imgtk  # keep reference
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

            self.logln("Enrolling: {}".format(name))
            embs = []
            captured = 0

            for i in range(ENROLL_IMAGES):
                prompt = PROMPTS[i] if i < len(PROMPTS) else "Pose"
                self.prompt_label.config(text="{}  [{} of {}]".format(prompt, i + 1, ENROLL_IMAGES))
                # Visible countdown on preview
                for tsec in range(CAPTURE_COUNTDOWN, 0, -1):
                    self._set_banner("{}\nCapturing in {}...".format(prompt, tsec), seconds=1.0)
                    self.update()
                    time.sleep(1.0)

                # Try several frames to find a face (steadier)
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

                face_path = os.path.join(person_dir, "face_{}.jpg".format(i + 1))
                try:
                    face_pil.save(face_path, quality=95)
                except Exception as e:
                    self.logln("  [!] Could not save {}: {}".format(face_path, e))
                    self._clear_banner()
                    continue

                try:
                    emb = pil_to_embedding(face_pil)
                    embs.append(emb)
                    captured += 1
                    self.logln("  Saved {}".format(face_path))
                except Exception as e:
                    self.logln("  [!] Embedding failed for sample {}: {}".format(i + 1, e))

                # Small pause after each capture so user can reset pose
                self.update()
                time.sleep(CAPTURE_PAUSE_AFTER)
                self._clear_banner()

            # Clear big prompt at end
            self.prompt_label.config(text="")

            if captured == 0:
                self.logln("[X] Enrollment failed: no valid face samples.")
                return

            try:
                out_path = save_signature(name, embs)
            except Exception as e:
                self.logln("[X] Could not save signature: {}".format(e))
                return

            self.db = load_signatures()
            self.refresh_listbox()
            self.logln("[OK] Signature saved: {} (samples: {})".format(out_path, captured))

        except Exception as e:
            self.logln("[!] Error during enrollment: {}".format(e))

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

            # Optional banner
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
                # Big visible result
                self.result_label.config(text="Recognized: {}".format(name), fg="#00cc00")
                self.logln("[OK] Recognized: {} (sim={:.3f} >= {:.2f}) -> GREEN".format(name, sim, thr))
                ok, resp = hue_green(bridge_ip, user, light_id)
                if not ok:
                    self.logln("[Hue] Error: {}".format(resp))
            else:
                close = name if name else "-"
                self.result_label.config(text="Unknown", fg="#cc0000")
                self.logln("[X] Unknown (closest: {}, sim={:.3f} < {:.2f}) -> RED".format(close, sim, thr))
                ok, resp = hue_red(bridge_ip, user, light_id)
                if not ok:
                    self.logln("[Hue] Error: {}".format(resp))

            # non-blocking auto-off
            if self.off_timer and self.off_timer.is_alive():
                pass
            self.logln("[LIGHT] Turning off in {}s...".format(timeout_s))
            self.off_timer = threading.Timer(timeout_s, self._turn_off_hue_safe,
                                             args=(bridge_ip, user, light_id))
            self.off_timer.daemon = True
            self.off_timer.start()

            # Clear banner + clear result label after a few seconds
            self._clear_banner()
            self.after(5000, lambda: self.result_label.config(text=""))
            self.logln("Ready for next recognition.")
        except Exception as e:
            self.logln("[!] Error during recognition: {}".format(e))

    # Timer callback: turn Hue off safely
    def _turn_off_hue_safe(self, ip: str, user: str, light_id: str):
        try:
            ok = hue_off(ip, user, light_id, transition_sec=1.5)
            self.logln("[Hue] Off." if ok else "[Hue] Off may have failed.")
        except Exception as e:
            self.logln("[Hue] Off error: {}".format(e))

	
# ---------------------------- Run ----------------------------
if __name__ == "__main__":
    app = FaceGUI()
    try:
        app.mainloop()
    finally:
        app.cam.stop()

