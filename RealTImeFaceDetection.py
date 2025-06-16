import cv2 #opencs-python

haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #Loading the Algorithm

cam = cv2.VideoCapture(0) #initializing the camera id #0 is primary camera

while True :

    _,img = cam.read() #Reading the frame from camera

    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Converting color image to Grayscale image

    faces = haar_cascade.detectMultiScale(grayimg,1.3,4) #Getting Coordinates

    for (x,y,w,h) in faces :
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3) #Drawing the Rectangle

    cv2.imshow("Face Detection",img) #Displaying the frame

    key = cv2.waitKey(10)
    print(key)

    if key == 27 : #Escape key to exit
        break

cam.release()
cv2.destroyAllWindows()
