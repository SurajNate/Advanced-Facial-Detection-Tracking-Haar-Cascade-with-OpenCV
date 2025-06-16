
#Multi Scale Face Detection - @suraj_nate

import cv2

# Load the pre-trained Haar Cascade classifier for face detection
haar_cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read the image
img = cv2.imread("group photo.jpg")

# Convert image to grayscale (Haar cascades work better on grayscale images)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = haar_cascades.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow("Multi Scale Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# scaleFactor -  Parameter specifying how much the image size is reduced at each image scale
# minNeighbors - Parameter Specifying how many rectangle should have to retain it, neighbors each candidate

