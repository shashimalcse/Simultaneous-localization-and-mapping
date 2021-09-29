import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform



class OrbDetector(object):

    def __init__(self,K):
        self.K =K
        self.Kinv = np.linalg.inv(K)
        self.orb = cv2.ORB_create()
        self.bfm =  cv2.BFMatcher(cv2.NORM_HAMMING)
        self.GX = 16//2
        self.GY = 12//2
        self.last = None
    def add_ones(self,x):
        return np.concatenate([x,np.ones((x.shape[0],1))],axis=1)    
    def denormalize(self,pt):
        ret =  np.dot(self.K,np.array([pt[0],pt[1],1.0]))  
        return int(round(ret[0])),int(round(ret[1])) 
    def extract2(self,img):
        pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
        kps,des = self.orb.compute(img,kps)
        
        ret = []
        if self.last is not None:
            matches = self.bfm.knnMatch(des,self.last['des'], k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last["kps"][m.trainIdx].pt
                    ret.append((kp1,kp2))
        if len(ret) > 0:
            ret = np.array(ret)  
            ret[:,0,:] = np.dot(self.Kinv,self.add_ones(ret[:,0,:]).T).T[:,0:2]
            ret[:,1,:] = np.dot(self.Kinv,self.add_ones(ret[:,1,:]).T).T[:,0:2]
            model, inliers = ransac((ret[:,0],
                            ret[:,1]),EssentialMatrixTransform, min_samples=8,residual_threshold=0.01, max_trials=100)
            ret = ret[inliers]
        self.last = {"kps":kps,"des":des}
        return ret          

cap = cv2.VideoCapture('0.hevc')

i=0
points = []
F = 240
W =1920
H = 1080
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
orbDetector = OrbDetector(K)

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    matches = orbDetector.extract2(frame)
    if matches is not None: 
        for kp1,kp2 in matches:
            x1,y1 = orbDetector.denormalize(kp1)

            x2,y2 = orbDetector.denormalize(kp2)
            cv2.circle(frame,(x1,y1),color=(0,255,0),radius =3)
            cv2.line(frame,(x1,y1),(x2,y2),color=(0,0,255),thickness =1)
            points.append([x1,y1,i*100])  
        cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break
  i+=1  
points = np.array(points)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd])