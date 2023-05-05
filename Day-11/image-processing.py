# To run this file we have to run 'Debug Python File' 

import cv2
import matplotlib.pyplot as py
import numpy as np
from time import sleep

fd = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)
while True:
    flag, img = vid.read()
    if flag:
        # All Processing Code Come here

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = fd.detectMultiScale(
            img_gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (50,50),
        )

        np.random.seed(50)
        colors = np.random.randint(0,255, (len(faces), 3)).tolist()
        i = 0

        # th, img_bw = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        for x,y,w,h in faces:
            # x,y,w,h = (250, 200, 100, 100)
            # img_cropped = img[y:y+h, x:x+w, :]

            cv2.rectangle(
                # img, pt1=(x,y), pt2=(x+w,y+h), color=[0,0,255], thickness=8

                # for different faces, different color reactangle 
                img, pt1=(x,y), pt2=(x+w,y+h), color=colors[i], thickness=8

            )
            i += 1
 
        # cv2.imshow('Preview', img_gray)
        # cv2.imshow('Preview', img_bw)
        # cv2.imshow('Preview', img_cropped)
        cv2.imshow('Preview', img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        print('No Frames')

        break 
    sleep(0.1)
vid.release() 
cv2.destroyAllWindows()