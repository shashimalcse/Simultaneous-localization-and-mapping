import numpy as np
import cv2
from tracker import Tracker
     
cap = cv2.VideoCapture('0.hevc')
tracker = Tracker()
frame_id = 0
traj_img_size = 800
traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
half_traj_img_size = int(0.5*traj_img_size)
draw_scale = 1
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
        if(frame_id>2 and frame_id%4==0):
            x, y, z = tracker.traj3d_est[-1]
            print(x, y, z)
            x_true, y_true, z_true = tracker.traj3d_gt[-1]  
            draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
            true_x, true_y = int(draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
            cv2.circle(traj_img, (draw_x, draw_y), 1,(0, 255, 0), 5)
            cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)		
            cv2.imshow('Trajectory', traj_img)      
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    frame_id += 1
