import numpy as np
import cv2
     

class Tracker(object):
    def __init__(self):
        self.cur_frame = None
        self.prev_frame = None
        self.kps_ref, self.des_ref = None,None
        self.kps_cur, self.des_cur = None,None
        self.mask_match = None
        self.trueX = 0 
        self.trueY = 0 
        self.trueZ = 0
        self.cur_R = np.eye(3,3)
        self.cur_t = np.zeros((3,1))
        self.t0_est = None
        self.t0_gt = None
        self.traj3d_est = []  
        self.traj3d_gt = []
        self.poses = [] 
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
    def poseRt(self,R, t):
        ret = np.eye(4)
        ret[:3, :3] = R
        ret[:3, 3] = t
        return ret    
    def estimate_pose(self,kps_ref,kps_cur):
        E, self.mask_match = cv2.findEssentialMat(self.kps_cur, self.kps_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=0.1)
        _, R, t, mask = cv2.recoverPose(E, self.kps_cur, self.kps_cur, focal=1, pp=(0., 0.))   
        return R,t
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
        R, t = self.estimate_pose(kps_ref_matched, kps_cur_matched) 
        self.cur_t = self.cur_t + self.cur_R.dot(t) 
        self.cur_R = self.cur_R.dot(R) 
        if self.kps_ref.shape[0] < 2000:
            pts = cv2.goodFeaturesToTrack(np.mean(self.cur_frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
            kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
            self.kps_cur, self.des_cur = self.orb.compute(self.cur_frame,kps)
            self.kps_cur = np.array([x.pt for x in self.kps_cur], dtype=np.float32) 
        self.kps_ref = self.kps_cur
        self.des_ref = self.des_cur
        self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])
        self.t0_gt  = np.array([self.trueX, self.trueY, self.trueZ])
        if (self.t0_est is not None) and (self.t0_gt is not None):             
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]] 
            self.traj3d_est.append(p)
            pg = [self.trueX-self.t0_gt[0], self.trueY-self.t0_gt[1], self.trueZ-self.t0_gt[2]]
            self.traj3d_gt.append(pg)     
            self.poses.append(self.poseRt(self.cur_R, p))