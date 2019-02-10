import cv2
import os

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

face_id = input('\n Enter user id :  ')

print("\n Look at the camera, system is collecting dataset")
count = 0

video_capture = cv2.VideoCapture(0)

while(True):
    print "inside while"
    ret, img = video_capture.read()
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5);

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
         break

print("\n Dataset generated")
cam.release()
cv2.destroyAllWindows()