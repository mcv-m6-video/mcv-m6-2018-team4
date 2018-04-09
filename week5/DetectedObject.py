import numpy as np
import sys
sys.path.append('../')
from KalmanFilter import *
import cv2
import math

class detectedObject:

    def __init__(self, id, obj_coord):

        self.id = id
        self.visible = True
        minx = min(obj_coord[0])
        maxx = max(obj_coord[0])
        miny = min(obj_coord[1])
        maxy = max(obj_coord[1])

        self.bbox = np.array([minx, maxx, miny, maxy])
        self.center = [(self.bbox[1]-self.bbox[0])/2,(self.bbox[3]-self.bbox[2])/2]

        self.kalman = KalmanFilter(self.center)

    def updateBBox(self,bbox):
        self.bbox = bbox
        self.center = [(self.bbox[1]-self.bbox[0])/2,(self.bbox[3]-self.bbox[2])/2]
