import flwr as fl
import numpy as np
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
import os
import cv2
from PIL import Image
import time
from flwr.common import parameters_to_ndarrays



# Federated Server Strategy
class FaceRecognitionStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        print(f"🔄 Aggregating round {rnd} results...")
        if not results:
            return None, {}

        # Extract parameters from each client
       # print(parameters_to_ndarrays(results[0][1].parameters))
        embeddings = [np.array(parameters_to_ndarrays(res.parameters)) for _ , res in results]  # ✅ Correct unpacking

        # Compute Global Face Signature (Average of All Clients)
        global_signature = np.mean(embeddings, axis=0)
        np.save("global_face_signature.npy", global_signature)  
        print(f"✅ New Global Face Signature Computed!")

        return [global_signature], {}

# Start Federated Server
server_address = "10.19.4.71:8080"
fl.server.start_server(
    server_address=server_address, 
    strategy=FaceRecognitionStrategy(), 
    config=fl.server.ServerConfig(num_rounds=1)
)

# Load FaceNet Model for Inference
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to Capture and Process Face

def capture_face():
    cap = cv2.VideoCapture(0)  # Open Webcam
    if not cap.isOpened():
        print("❌ Error: Camera cannot be opened!")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        brightness_reduction = 25
        adjusted_image = cv2.subtract(frame, np.full(frame.shape, brightness_reduction, dtype=np.uint8))
        adjusted_image = cv2.flip(adjusted_image, 1)
        test_img_path = "captured_face.jpg"
        cv2.imwrite(test_img_path, adjusted_image)
        return test_img_path
    else:
        print("❌ Error: Image capture failed.")
        return None

# Function to Perform Face Recognition

def recognize_face(global_signature, threshold=0.6):
    test_img_path = capture_face()
    if test_img_path is None:
        return
    
    image = Image.open(test_img_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        test_embedding = model(image).squeeze().numpy()
    
    cosine_similarity = np.dot(global_signature, test_embedding) / (
        np.linalg.norm(global_signature) * np.linalg.norm(test_embedding)
    )
    
    if cosine_similarity > threshold:
        print(f"\n✅ Recognized! (Similarity: {cosine_similarity}) 🎉")
        save_img_path = f"recognized_{int(time.time())}.jpg"
        os.rename(test_img_path, save_img_path)
        print(f"✅ Image saved as {save_img_path}")
    else:
        print(f"\n❌ Not Recognized! (Similarity: {cosine_similarity})")

# Periodic Inference Loop
if os.path.exists("global_face_signature.npy"):
    global_face_signature = np.load("global_face_signature.npy")
    print("✅ Loaded global face signature for real-time inference.")
    while True:
        recognize_face(global_face_signature)
        time.sleep(1)  # Run inference every 10 seconds
else:
    print("❌ Global face signature not found. Run federated learning first.")
