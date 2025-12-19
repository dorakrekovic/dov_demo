import cv2
import numpy as np
import os

# 🎥 Open the camera
cap = cv2.VideoCapture(0)  # Change to 1 if the wrong camera is used

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 📸 Capture an image
ret, frame = cap.read()
cap.release()

if ret:
    # 🛠 Adjust Brightness (Reduce by 50 - Adjust as needed)
    brightness_reduction = 25
    adjusted_image = cv2.subtract(frame, np.full(frame.shape, brightness_reduction, dtype=np.uint8))
    adjusted_image =  cv2.flip(adjusted_image, 1)
    # 💾 Save the adjusted image
    adjusted_image_path = "adjusted_image.jpg"
    cv2.imwrite(adjusted_image_path, adjusted_image)

    print(f"✅ Image saved as {adjusted_image_path}")

    # 📤 Transfer the image to your PC using direct IP
    destination = "valentin@10.19.4.157:/home/valentin/Desktop/"
    scp_command = f"scp {adjusted_image_path} {destination}"
    os.system(scp_command)

    print("🚀 Image transferred successfully to your PC!")

else:
    print("Error: Could not capture image.")

