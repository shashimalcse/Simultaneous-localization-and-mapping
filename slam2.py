import numpy as np
import cv2
from tracker import Tracker
     
cap = cv2.VideoCapture('0.hevc')
tracker = Tracker()
frame_id = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        tracker.track(frame,frame_id)
        if (frame_id!=0):  
            for i,pts in enumerate(zip(tracker.kps_ref, tracker.kps_cur)):
                p1, p2 = pts 
                a,b = p1.astype(int).ravel()
                c,d = p2.astype(int).ravel()
                cv2.line(frame, (a,b),(c,d), (0,0,255), 2)
                cv2.circle(frame,(c,d),1, (0,255,),4)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    frame_id += 1
