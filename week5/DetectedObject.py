import numpy as np
import sys
sys.path.append('../')
from KalmanFilter import *
import cv2
import math

class detectedObject:

    def __init__(self, id, obj_coord, nFrame):

        self.id = id
        self.nFrame = nFrame
        self.visible = True
        self.finished = False

        minx = min(obj_coord[0])
        maxx = max(obj_coord[0])
        miny = min(obj_coord[1])
        maxy = max(obj_coord[1])

        self.bbox = np.array([minx, maxx, miny, maxy])
        self.center = [(self.bbox[1]-self.bbox[0])/2+self.bbox[0], (self.bbox[3]-self.bbox[2])/2+self.bbox[2]]

        self.posList = [self.center]
        self.bboxList = [self.bbox]
        self.framesList = [self.nFrame]
        self.velocities = [50]

        self.kalman = KalmanFilter(self.center)

    def updateBBox(self,bbox, nFrame):
        self.bbox = bbox
        self.center = [(self.bbox[1]-self.bbox[0])/2+self.bbox[0], (self.bbox[3]-self.bbox[2])/2+self.bbox[2]]

        self.posList.append(self.center)
        self.bboxList.append(bbox)
        self.framesList.append(nFrame)

    def updateVelocity(self, velocity, frame_step):
        for i in range(frame_step):
            self.velocities.append(velocity)

