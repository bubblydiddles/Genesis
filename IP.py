import cv2
import numpy as np
import time

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 200
params.maxThreshold = 255
    
params.filterByArea = True
params.minArea = 150
params.maxArea = 1000
    
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)

cam = cv2.VideoCapture(0)

ll = np.array([88,153,90])
ul = np.array([130,255,255])

def nothing(x):
    pass

cv2.namedWindow('res')

kernel = np.ones((3,3))

#cv2.createTrackbar('lh', 'res', 0, 180, nothing)
#cv2.createTrackbar('uh', 'res', 0, 180, nothing)
#cv2.createTrackbar('ls', 'res', 0, 255, nothing)
#cv2.createTrackbar('us', 'res', 0, 255, nothing)
#cv2.createTrackbar('lv', 'res', 0, 255, nothing)
#cv2.createTrackbar('uv', 'res', 0, 255, nothing)

while True:
    time.sleep(100/1000)
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, ll, ul)
    res = cv2.bitwise_or(frame,frame, mask=mask)

    keypoints = detector.detect(mask)
    res2 = cv2.drawKeypoints(res, keypoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow('res' , res2)
    #lh = cv2.getTrackbarPos('lh', 'res')
    #uh = cv2.getTrackbarPos('uh', 'res')
    #ls = cv2.getTrackbarPos('ls', 'res')
    #us = cv2.getTrackbarPos('us', 'res')
    #lv = cv2.getTrackbarPos('lv', 'res')
    #uv = cv2.getTrackbarPos('uv', 'res')
    
    #ll = np.array([lh,ls,lv])
    #ul = np.array([uh,us,uv])

    for i in keypoints:
        x=i.pt[0]; y=i.pt[1]
        print(x,y)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
