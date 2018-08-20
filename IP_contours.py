import cv2
import numpy as np
import time
import serial


cam = cv2.VideoCapture(0)

ll = np.array([0,176,109])
ul = np.array([15,255,255])

def nothing(x):
    pass

cv2.namedWindow('Isolated Contour')

flag = 0

kernel = np.ones((3,3))

#cv2.createTrackbar('lh', 'Isolated Blob', 0, 180, nothing)
#cv2.createTrackbar('uh', 'Isolated Blob', 0, 180, nothing)
#cv2.createTrackbar('ls', 'Isolated Blob', 0, 255, nothing)
#cv2.createTrackbar('us', 'Isolated Blob', 0, 255, nothing)
#cv2.createTrackbar('lv', 'Isolated Blob', 0, 255, nothing)
#cv2.createTrackbar('uv', 'Isolated Blob', 0, 255, nothing)

while True:
    time.sleep(10/1000)
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, ll, ul)
    res = cv2.bitwise_or(frame,frame, mask=mask)

    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    maxarea = 0

    try:
        c = max(contours, key = cv2.contourArea)
        if cv2.contourArea(c) > 10:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)
            px = x + (w/2)
            py = y + (h/2)
            print(px,py)

    except:
        pass

    cv2.imshow('Live Feed',frame)
    cv2.imshow('Isolated Contour' , res)
    
    #lh = cv2.getTrackbarPos('lh', 'Isolated Blob')
    #uh = cv2.getTrackbarPos('uh', 'Isolated Blob')
    #ls = cv2.getTrackbarPos('ls', 'Isolated Blob')
    #us = cv2.getTrackbarPos('us', 'Isolated Blob')
    #lv = cv2.getTrackbarPos('lv', 'Isolated Blob')
    #uv = cv2.getTrackbarPos('uv', 'Isolated Blob')
    
    #ll = np.array([lh,ls,lv])
    #ul = np.array([uh,us,uv])

    if cv2.waitKey(1) & 0xFF == ord(' '):
        flag = 1
    else:
        flag = 0
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

