import cv2
import numpy as np
import time

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 200
params.maxThreshold = 255
    
params.filterByArea = True
params.minArea = 100
params.maxArea = 30720
    
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)

cam = cv2.VideoCapture(0)

ll = np.array([0,172,126])
ul = np.array([15,255,225])

def nothing(x):
    pass

bg = np.ones((640,480))

cv2.namedWindow('Isolated Blob')
cv2.namedWindow('Plot')

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

    keypoints = detector.detect(mask)
    res2 = cv2.drawKeypoints(res, keypoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Live Feed',frame)
    cv2.imshow('Isolated Blob' , res2)
    
    #lh = cv2.getTrackbarPos('lh', 'Isolated Blob')
    #uh = cv2.getTrackbarPos('uh', 'Isolated Blob')
    #ls = cv2.getTrackbarPos('ls', 'Isolated Blob')
    #us = cv2.getTrackbarPos('us', 'Isolated Blob')
    #lv = cv2.getTrackbarPos('lv', 'Isolated Blob')
    #uv = cv2.getTrackbarPos('uv', 'Isolated Blob')
    
    #ll = np.array([lh,ls,lv])
    #ul = np.array([uh,us,uv])

    if flag == 0:
        for i in keypoints:
            x=int(i.pt[0]); y=int(i.pt[1])
            print(x,y)
            cv2.imshow('Plotted Points', bg)
            bg[y,x] = 0

    if cv2.waitKey(1) & 0xFF == ord(' '):
        flag = 1
    else:
        flag = 0
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
