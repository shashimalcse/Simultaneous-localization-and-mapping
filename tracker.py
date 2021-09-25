import numpy as np
import cv2
     

class Tracker(object):
    def __init__(self):
        self.cur_frame = None
        self.prev_frame = None
        self.kps_ref, self.des_ref = None,None
        self.kps_cur, self.des_cur = None,None
        self.orb = cv2.ORB_create()
        self.lk_params = dict(winSize  = (21, 21), 
                              maxLevel = 3,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))        
    def track(self, frame,frame_id):
        self.cur_frame = frame
        if(frame_id==0):
            self.process_first_frame()
        else:
            self.process_frame(frame_id)    
        self.prev_frame = frame
    def process_first_frame(self):   
        pts = cv2.goodFeaturesToTrack(np.mean(self.cur_frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
        self.kps_ref, self.des_ref = self.orb.compute(self.cur_frame,kps)
        self.kps_ref = np.array([x.pt for x in self.kps_ref], dtype=np.float32) 
    def process_frame(self,frame_id):
        kps_cur, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, self.cur_frame, self.kps_ref,None,**self.lk_params)
        idx = [i for i,v in enumerate(st) if v== 1]
        kps_ref_matched = self.kps_ref[idx] 
        kps_cur_matched = kps_cur[idx.copy()]  
        self.kps_ref = kps_ref_matched
        self.kps_cur = kps_cur_matched
        if self.kps_ref.shape[0] < 2000:
            pts = cv2.goodFeaturesToTrack(np.mean(self.cur_frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
            kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
            self.kps_cur, self.des_cur = self.orb.compute(self.cur_frame,kps)
            self.kps_cur = np.array([x.pt for x in self.kps_cur], dtype=np.float32) 
        self.kps_ref = self.kps_cur
        self.des_ref = self.des_cur