import torch
import torch.nn as nn
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
import os
import numpy as np
from PIL import Image
import cv2

data_folder = os.path.expanduser("./myfaces")  
model = InceptionResnetV1(pretrained='vggface2').eval()  # Load Pretrained FaceNet Model

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_face_embedding(img_path):
    image = Image.open(img_path).convert("RGB")  # Load and convert to RGB
    image = transform(image).unsqueeze(0)  # Apply transformations
    with torch.no_grad():
        embedding = model(image)  # Get the embedding
    return embedding.squeeze().numpy()

#  Step 1: Training (Press ENTER to start)
input(" Press ENTER to start training on your face images...")
face_embeddings = []

for filename in os.listdir(data_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(data_folder, filename)
        print(f" Learning from {filename}...")
        face_embeddings.append(get_face_embedding(img_path))


if face_embeddings:
    your_face_signature = np.mean(face_embeddings, axis=0)
    np.save("your_face_signature.npy", your_face_signature) 
    print("Your face signature learned and saved!")
else:
    print("No images found in 'myfaces/' folder.")
    exit()

# Step 2: Inference (Press ENTER to capture a new face image)
input("\n Press ENTER to capture a new face image for recognition...")
cap = cv2.VideoCapture(0)  # Open Webcam

if not cap.isOpened():
    print(" Error: Camera cannot be opened!")
    exit()

print(" Capturing image... Look at the camera!")
ret, frame = cap.read()
cap.release()

if ret:
    # Adjust brightness and flip horizontally
    #brightness_reduction = 25
    #adjusted_image = cv2.subtract(frame, np.full(frame.shape, brightness_reduction, dtype=np.uint8))
    #adjusted_image = cv2.flip(adjusted_image, 1)

    # Save the captured image
    test_img_path = "test_face.jpg"
    cv2.imwrite(test_img_path, frame)
    print(f" Image saved as {test_img_path}")

    # Extract embedding for the new image
    image = Image.open(test_img_path).convert("RGB")  # Convert to RGB
    image = transform(image).unsqueeze(0)  # Apply transformations
    with torch.no_grad():
        test_embedding = model(image).squeeze().numpy()

    # Compute similarity (Cosine Similarity)
    cosine_similarity = np.dot(your_face_signature, test_embedding) / (np.linalg.norm(your_face_signature) * np.linalg.norm(test_embedding))

    #  Decision: Recognized or Not?
    threshold = 0.6  # Adjust if needed
    if cosine_similarity > threshold:
        print(f"\n Recognized! (Similarity: {cosine_similarity:.2f}) 🎉")
    else:
        print(f"\n Not recognized! (Similarity: {cosine_similarity:.2f})")

else:
    print("Error: Image capture failed.")

