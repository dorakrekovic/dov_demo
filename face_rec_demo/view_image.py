import cv2

# Load the image
image = cv2.imread("test_face.jpg")

# Show the image in a new window
cv2.imshow("Captured Image", image)
cv2.waitKey(0)  # Wait for key press to close
cv2.destroyAllWindows()

