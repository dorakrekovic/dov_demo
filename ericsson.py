import time
import paho.mqtt.client as mqtt
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import json
import requests
from requests.auth import HTTPBasicAuth
import datetime


# MQTT Settings
BROKER = 'djx.entlab.hr'  
PORT = 1883
USERNAME = 'digiphy2'
PASSWORD = 'pTWBAkrMplwnQ3Ft'
SUB_TOPIC = 'iotlab/rpi4/capture/5f48d1f7-20ae-4ebd-a95e-9afa968ddfe2'
PUB_TOPIC = 'detection/result'
CLIENT_ID = '5f48d1f7-20ae-4ebd-a95e-9afa968ddfe2'

# Flag to trigger the face recognition
start_face_recognition = False

# Load Pretrained FaceNet Model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the saved face signature
face_signature_path = "your_face_signature.npy"
if os.path.exists(face_signature_path):
    your_face_signature = np.load(face_signature_path)
    print("Your face signature loaded successfully!")
else:
    print(f"Error: '{face_signature_path}' not found. Make sure to train first.")
    exit()

def get_face_embedding(img_path):
    """Function to get face embedding from an image."""
    image = Image.open(img_path).convert("RGB")  # Load and convert to RGB
    image = transform(image).unsqueeze(0)  # Apply transformations
    with torch.no_grad():
        embedding = model(image)  # Get the embedding
    return embedding.squeeze().numpy()

def on_connect(client, userdata, flags, rc):
    print("Connected to broker with code:", rc)
    client.subscribe(SUB_TOPIC)

def on_message(client, userdata, msg):
    global start_face_recognition
    print(datetime.datetime.now())
    print(f"Message received on topic {msg.topic}: {msg.payload.decode()}")

    if msg.topic == SUB_TOPIC:
        start_face_recognition = True

def capture_and_recognize_face():
    """Capture a face image from the webcam and recognize it."""
    print("Capturing image... Look at the camera!")
    cap = cv2.VideoCapture(0)  # Open Webcam

    if not cap.isOpened():
        print("Error: Camera cannot be opened!")
        return None

    ret, frame = cap.read()
    cap.release()

    if ret:
        # Save the captured image
        test_img_path = "test_face.jpg"
        cv2.imwrite(test_img_path, frame)
        print(f"Image saved as {test_img_path}")

        # Extract embedding for the new image
        image = Image.open(test_img_path).convert("RGB")  # Convert to RGB
        image = transform(image).unsqueeze(0)  # Apply transformations
        with torch.no_grad():
            test_embedding = model(image).squeeze().numpy()

        # Compute similarity (Cosine Similarity)
        cosine_similarity = np.dot(your_face_signature, test_embedding) / (np.linalg.norm(your_face_signature) * np.linalg.norm(test_embedding))

        # Decision: Recognized or Not?
        threshold = 0.6  # Adjust if needed
        if cosine_similarity > threshold:
            print(f"Recognized! (Similarity: {cosine_similarity:.2f})")
            result_message= "true"
        else:
            print(f"Not recognized! (Similarity: {cosine_similarity:.2f})")
            result_message= "false"

        url = "https://djx.entlab.hr/m2m/data"
        headers = {
            "Content-Type": "application/vnd.ericsson.m2m.input+json"
        }
        data = {
            "contentNodes": [
                {
                    "source": {
                        "resource": "4a65f963-f757-45dc-b45d-75e124fde261"
                    },
                    "value": result_message
                }
            ]
        }

        response = requests.post(
            url,
            auth=HTTPBasicAuth("digiphy2", "pTWBAkrMplwnQ3Ft"),
            headers=headers,
            json=data
        )

        print(response.status_code)
        print(response.text)

    else:
        print("Error: Image capture failed.")
        return None
       

def main():
    global start_face_recognition

    # MQTT Client Setup
    client = mqtt.Client(CLIENT_ID)
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)
    client.loop_start()

    print(f"Waiting for 'start' command on topic: {SUB_TOPIC}")

    try:
        while True:
            if start_face_recognition:
                # Perform face recognition
                result_message = capture_and_recognize_face()

                if result_message:
                    # Publish the result in the desired JSON format
                    client.publish(PUB_TOPIC, result_message)
                    print(f"Published to {PUB_TOPIC}: {result_message}")

                # Reset flag
                start_face_recognition = False

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
