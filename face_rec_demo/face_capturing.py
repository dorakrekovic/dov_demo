import cv2
import os
import numpy as np

#  Create a folder on Desktop to store adjusted images
save_path = os.path.expanduser("./myfaces")  # Path to save images
os.makedirs(save_path, exist_ok=True)

#  Open the webcam
cap = cv2.VideoCapture(0)  # Change to 1 if needed

if not cap.isOpened():
    print(" Greška: Kamera se ne može otvoriti!")
    exit()

expressions = [
    "Pogledajte ravno u kameru s neutralnim izrazom. (Pritisnite ENTER za snimanje)",
    "Okrenite lice malo ulijevo. (Pritisnite ENTER za snimanje)",
    "Okrenite lice malo udesno. (Pritisnite ENTER za snimanje)",
    "Pogledajte malo prema gore. (Pritisnite ENTER za snimanje)",
    "Pogledajte malo prema dolje. (Pritisnite ENTER za snimanje)",
    "Nasmiješite se. (Pritisnite ENTER za snimanje)",
    "Napravite ozbiljan izraz lica. (Pritisnite ENTER za snimanje)",
    "Namignite jednim okom. (Pritisnite ENTER za snimanje)"
]

for i, expression in enumerate(expressions):
    input(f"\n {expression}")  # Wait for ENTER key

    ret, frame = cap.read()  # Capture frame
    if ret:
        # Adjust Brightness (Reduce by 25 for balance)
        brightness_reduction = 25
        adjusted_image = cv2.subtract(frame, np.full(frame.shape, brightness_reduction, dtype=np.uint8))

        #  Flip Image Horizontally
        adjusted_image = cv2.flip(adjusted_image, 1)

        #  Save the adjusted image to Desktop folder
        adjusted_image_path = os.path.join(save_path, f"face_{i+1}.jpg")
        cv2.imwrite(adjusted_image_path, adjusted_image)

        print(f" Slika spremljena: {adjusted_image_path}")

        #  Transfer the image to PC via SCP
        #destination = "valentin@10.19.4.157:/home/valentin/Desktop/"
        #scp_command = f"scp {adjusted_image_path} {destination}"
        #os.system(scp_command)

        #print(" Slika uspješno poslana na vaše računalo!")

    else:
        print(" Greška: Neuspješno snimanje slike.")

cap.release()
cv2.destroyAllWindows()

print("\n Sve slike su snimljene, spremljene u 'myfaces' i poslane na vaše računalo! 🚀")

