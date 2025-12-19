# enroll_face.py
import os
import sys
import time
import json
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

# ───────────────────────────── Settings ─────────────────────────────
OUTPUT_DATASET_DIR = os.path.expanduser("./dataset")     # cropped training images per person
OUTPUT_SIGNATURE_DIR = os.path.expanduser("./signatures")# embeddings per person
INDEX_PATH = os.path.join(OUTPUT_SIGNATURE_DIR, "index.json")

# How many images to capture per person
NUM_IMAGES = 8

# Inference normalization & threshold aren’t used here, but we normalize the signature anyway
# to keep things consistent and robust later.
# ───────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_SIGNATURE_DIR, exist_ok=True)

# Model + transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

PROMPTS = [
    "Pogledajte ravno u kameru (ENTER) / Look straight at the camera (ENTER)",
    "Okrenite lice malo ulijevo (ENTER) / Turn slightly left (ENTER)",
    "Okrenite lice malo udesno (ENTER) / Turn slightly right (ENTER)",
    "Pogledajte malo prema gore (ENTER) / Look slightly up (ENTER)",
    "Pogledajte malo prema dolje (ENTER) / Look slightly down (ENTER)",
    "Nasmiješite se (ENTER) / Smile (ENTER)",
    "Ozbiljan izraz (ENTER) / Neutral-serious (ENTER)",
    "Opustite lice (ENTER) / Relax face (ENTER)",
]

def load_index():
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_index(idx):
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec) + 1e-8
    return vec / n

def get_embedding_from_pil(pil_img: Image.Image) -> np.ndarray:
    # Transform and embed
    img_t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(img_t).squeeze().cpu().numpy().astype(np.float32)
    return l2_normalize(emb)

def capture_and_crop_face(cap: cv2.VideoCapture):
    """
    Capture a frame, detect & align face with MTCNN, return aligned PIL image (160x160).
    Returns (pil_face, bgr_frame_shown) or (None, frame) on failure.
    """
    ok, frame = cap.read()
    if not ok:
        return None, None

    # Convert BGR -> RGB PIL for MTCNN
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # Detect & align
    face = mtcnn(pil)  # returns tensor CHW in [-1,1] if post_process=True; but we’ll re-run our transform anyway
    if face is None:
        return None, frame

    # Convert the returned tensor back to PIL, then re-apply our transform pipeline
    face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    face_pil = Image.fromarray(face_np)
    return face_pil, frame

def enroll_person(person_name: str, num_images: int = NUM_IMAGES):
    person_name = person_name.strip()
    if not person_name:
        print("Greška: Ime osobe je prazno.")
        sys.exit(1)

    person_dir = os.path.join(OUTPUT_DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"\n➡️  Započinjemo snimanje za: {person_name}")
    print("➡️  Starting capture for:", person_name)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Greška: Kamera se ne može otvoriti! / Error: Cannot open camera.")
        sys.exit(1)

    # Let exposure settle a bit
    time.sleep(0.3)
    collected = 0
    embeddings = []

    for i in range(num_images):
        prompt = PROMPTS[i] if i < len(PROMPTS) else f"Pozicija {i+1} (ENTER) / Pose {i+1} (ENTER)"
        input(f"\n{prompt}")

        # try up to a few times to get a face
        attempts = 0
        face_pil = None
        last_frame = None
        while attempts < 5 and face_pil is None:
            face_pil, last_frame = capture_and_crop_face(cap)
            if face_pil is None:
                attempts += 1
                print("Nije pronađeno lice, pokušavam ponovo... / No face found, retrying...")
                time.sleep(0.2)

        if face_pil is None:
            print("Preskačem ovu sliku (nema lica). / Skipping this one (no face).")
            continue

        # Save cropped face image (for audit/visual check)
        save_path = os.path.join(person_dir, f"face_{i+1}.jpg")
        face_pil.save(save_path, quality=95)
        print(f"Spremljeno: {save_path}")

        # Compute embedding
        emb = get_embedding_from_pil(face_pil)
        embeddings.append(emb)
        collected += 1

    cap.release()
    cv2.destroyAllWindows()

    if collected == 0:
        print("\nNije zabilježena nijedna ispravna slika. / No valid images captured.")
        sys.exit(1)

    # Mean embedding (then L2-normalize again)
    signature = np.mean(embeddings, axis=0).astype(np.float32)
    signature = l2_normalize(signature)

    # Save signature
    sig_path = os.path.join(OUTPUT_SIGNATURE_DIR, f"{person_name}.npy")
    np.save(sig_path, signature)
    print(f"\n✅ Potpis lica spremljen: {sig_path}")

    # Update index
    idx = load_index()
    idx[person_name] = {
        "signature_path": sig_path,
        "images": collected,
        "dataset_dir": person_dir,
        "created_at": datetime.now().isoformat(timespec="seconds")
    }
    save_index(idx)
    print(f"📘 Ažuriran indeks: {INDEX_PATH}")

    print("\nGotovo! / Done! 🎉")
    print("Ovaj potpis sada možete koristiti u skripti za prepoznavanje i Hue kontrolu.")

def main():
    if len(sys.argv) >= 2:
        name = " ".join(sys.argv[1:])
    else:
        name = input("Unesite ime i prezime kandidata / Enter candidate full name: ").strip()

    enroll_person(name, NUM_IMAGES)

if __name__ == "__main__":
    main()

