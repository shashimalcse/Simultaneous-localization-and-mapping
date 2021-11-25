import numpy as np
import cv2

class Tracker(object):
    def __init__(self):
        self.cur_frame = None
        self.prev_frame = None
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.cur_kps,self.cur_des = None,None
        self.ref_kps,self.ref_des = None,None
        self.ref_matches,self.cur_matches = None,None
        self.rmat = np.eye(3)
        self.tvec = np.zeros((3, 1))
    def track(self,frame,frame_id):
        self.cur_frame = frame
        if (frame_id==0):
            self.process_first_frame(frame)
        else:
            self.process_frame(frame)
        self.prev_frame = frame
    def process_first_frame(self,frame):
        pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
        self.ref_kps,self.ref_des = self.orb.compute(frame,kps)
    def estimate_motion(self):
        frame1_points = []
        frame2_points = []
        for i in range(len(self.cur_matches)):
            u1, v1 = self.cur_matches[i]
            u2, v2 = self.ref_matches[i]
            p_c = np.linalg.inv(self.k) @ (s * np.array([u1, v1, 1]))

    def process_frame(self,frame):
        pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
        self.cur_kps,self.cur_des = self.orb.compute(frame,kps)
        matches = self.matcher.knnMatch(self.cur_des,self.ref_des,k=2)
        query_matches = []
        train_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                query_matches.append(self.cur_kps[m.queryIdx].pt)
                train_matches.append(self.ref_kps[m.trainIdx].pt)
        self.ref_matches = np.float32(query_matches)
        self.cur_matches = np.float32(train_matches)
        self.ref_kps = self.cur_kps
        self.ref_des = self.cur_des
cap = cv2.VideoCapture('slam/0.hevc')
tracker = Tracker()
frame_id = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        tracker.track(frame,frame_id)
        if(frame_id>2):
            for i,pts in enumerate(zip(tracker.ref_matches, tracker.cur_matches)):
                p1, p2 = pts
                a,b = p1
                a,b = int(a),int(b)
                c,d = p2
                c,d = int(c),int(d)
                cv2.line(frame, (a,b),(c,d), (0,0,255), 2)
                cv2.circle(frame,(c,d),1, (0,255,),4)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    frame_id += 1
       
