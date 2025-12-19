import time
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import glob
import requests

# ── Philips Hue Settings (EDIT THESE) ──────────────────────────────────────────
HUE_BRIDGE_IP = "10.19.4.128"        # ← set your bridge IP
HUE_USERNAME  = "v84V00gmP09XseQaCEnqkIKT68yUXm8uLD8s9LJi"    # ← set your Hue API key (username)
HUE_LIGHT_ID  = "4"                    # ← set the light ID (string or int)

# ── Recognition settings ───────────────────────────────────────────────────────
SIGNATURE_DIR = "./signatures"   # where <Name>.npy are stored
THRESHOLD = 0.20                 # cosine similarity threshold
LIGHT_TIMEOUT = 10               # seconds to keep light on

# ── Model + transforms ────────────────────────────────────────────────────────
model = InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ── Hue helpers ───────────────────────────────────────────────────────────────
import requests, time

def _hue_put(url, payload, tries=3, delay=0.3):
    for i in range(tries):
        try:
            r = requests.put(url, json=payload, timeout=3)
            print(f"[Hue] PUT {url} -> {r.status_code} {r.text[:120]}")
            r.raise_for_status()
            return True
        except Exception as e:
            print(f"[Hue] Attempt {i+1} failed: {e}")
            time.sleep(delay)
    return False

def hue_set_state(payload: dict) -> bool:
    url = f"http://{HUE_BRIDGE_IP}/api/{HUE_USERNAME}/lights/{HUE_LIGHT_ID}/state"
    return _hue_put(url, payload)

def hue_green():
    # cancel effects, full bright, set green
    return hue_set_state({"on": True, "effect": "none", "alert": "none",
                          "hue": 25500, "sat": 254, "bri": 254})

def hue_red():
    # cancel effects, full bright, set red
    return hue_set_state({"on": True, "effect": "none", "alert": "none",
                          "hue": 0, "sat": 254, "bri": 254})

def hue_off_smooth(transition_seconds=2.0):
    """
    Smoothly fade then turn off. Also tries group 0 as a fallback in case a scene/automation re-applies state.
    """
    transitiontime = max(1, int(transition_seconds * 10))  # Hue units = 1/10s

    # 1) cancel effects and dim down
    ok1 = hue_set_state({"effect": "none", "alert": "none", "bri": 1, "transitiontime": transitiontime})
    # 2) turn off with a short transition
    ok2 = hue_set_state({"on": False, "transitiontime": transitiontime})

    # 3) Fallback: try groups/0 (all lights) if individual off failed or is instantly overridden
    if not (ok1 and ok2):
        group_url = f"http://{HUE_BRIDGE_IP}/api/{HUE_USERNAME}/groups/0/action"
        _hue_put(group_url, {"on": False, "transitiontime": transitiontime})

    return ok1 and ok2


# ── Utils ─────────────────────────────────────────────────────────────────────
def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)

def load_all_signatures(sig_dir: str):
    """Return dict: {name: normalized_embedding} from all .npy files."""
    if not os.path.isdir(sig_dir):
        print(f"❌ Signature folder '{sig_dir}' not found.")
        return {}
    db = {}
    for path in glob.glob(os.path.join(sig_dir, "*.npy")):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            vec = np.load(path).astype(np.float32)
            db[name] = l2_normalize(vec)
        except Exception as e:
            print(f"Skipping '{path}': {e}")
    return db

def frame_to_embedding(frame_bgr):
    """Convert a BGR frame to normalized embedding."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    img = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img).squeeze().numpy().astype(np.float32)
    return l2_normalize(emb)

def capture_frame(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Camera cannot be opened!")
        return None
    # let exposure settle a bit
    _ = cap.read()
    time.sleep(0.15)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Failed to capture frame.")
        return None
    return frame

def recognize_best(emb, db):
    """Return (best_name, best_similarity)."""
    best_name, best_sim = "", float("-inf")
    for name, sig in db.items():
        sim = float(np.dot(sig, emb))  # cosine (both L2-normalized)
        if sim > best_sim:
            best_sim, best_name = sim, name
    return best_name, best_sim

# ── Main loop ─────────────────────────────────────────────────────────────────
def run_once(db):
    if not db:
        print(f"❌ No signatures loaded from '{SIGNATURE_DIR}'. Enroll first.")
        return
    print("\n📸 Capturing image... Look at the camera!")
    frame = capture_frame(0)
    if frame is None:
        return
    emb = frame_to_embedding(frame)
    best_name, best_sim = recognize_best(emb, db)
    recognized = best_sim >= THRESHOLD

    if recognized:
        print(f"✅ Recognized: {best_name} (similarity={best_sim:.3f})")
        print("[Hue] GREEN")
        hue_green()
    else:
        print(f"❌ Unknown (closest: {best_name or '—'} | similarity={best_sim:.3f})")
        print("[Hue] RED")
        hue_red()

    print(f"💡 Turning off in {LIGHT_TIMEOUT}s...")
    time.sleep(LIGHT_TIMEOUT)
    if hue_off_smooth(transition_seconds=2.0):
    	print("[Hue] Off.")
    else:
    	print("[Hue] Off command may have failed (see logs above).")


def main():
    print("🔎 Loading signatures...")
    db = load_all_signatures(SIGNATURE_DIR)
    print(f"Loaded {len(db)} signature(s): {', '.join(sorted(db.keys())) or '—'}")

    print("\nPress ENTER for the next recognition (Ctrl+C to quit).")
    try:
        while True:
            input()
            run_once(db)
    except KeyboardInterrupt:
        print("\n👋 Bye.")

if __name__ == "__main__":
    main()

