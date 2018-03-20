
import sys
sys.path.append('..')
import cv2
import evaluation as ev
import numpy as np
import os
import time

from lucas_kanade import *
from horn_schunk import *
from block_matching import block_matching

from dataset import Dataset
from sklearn.metrics import auc

import sys
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import ndimage



def main():

    # Read datasets
    # TRAFFIC
    # traffic_dataset = Dataset('traffic',951, 1050)
    #
    # traffic = traffic_dataset.readInput()
    # traffic_GT = traffic_dataset.readGT()
    #
    # # Split dataset
    # traffic_train = traffic[:len(traffic)/2]
    # traffic_test = traffic[len(traffic)/2:]
    # traffic_test_GT = traffic_GT[len(traffic)/2:]

    # OPTICAL FLOW
    path = '../../Datasets/data_stereo_flow/training/'
    nsequence = '000045'
    # nsequence = '000157'

    F_gt = flow_read(os.path.join(path, 'flow_noc', nsequence+'_10.png'))

    sequence = []
    frame = cv2.imread(os.path.join(os.path.join(path,'image_0',nsequence+'_10.png')))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sequence.append(frame)
    frame = cv2.imread(os.path.join(os.path.join(path,'image_0',nsequence+'_11.png')))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sequence.append(frame)


# TASK 1 - OPTICAL FLOW
# TASK 1.1 - Block Matching
    # flow = block_matching(sequence[0], sequence[1], block_size=(16, 16), area=(2*16 +16, 2 *16+16))

    # rgb_flow = optical_flow_visualization(flow[:,:,0],flow[:,:,1])

    # plt.imshow(rgb_flow)
    # plt.show()

# TASK 1.2 - Block Matching vs Other Techniques
    sigma = 15;
    u, v = optical_flow_lk(sequence[0],sequence[1],sigma)

    # alpha = 0.5;
    # u, v = optical_flow_hs(sequence[0],sequence[1],alpha)

    rgb_flow = optical_flow_visualization(u,v)

    # flow = cv2.calcOpticalFlowFarneback(sequence[0], sequence[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # rgb_flow = optical_flow_visualization(flow[:,:,0],flow[:,:,1],0)
    # rgb_flow = optical_flow_visualization(F_gt[:,:,0],F_gt[:,:,1],3)

    plt.imshow(rgb_flow)
    plt.show()



# TASK 2 - VIDEO STABILIZATION
    # TASK 2.1 - Video stabilization with Block Matching

    # TASK 2.2 - Block Matching Stabilization vs Other Techniques

    # TASK 2.3 - Stabilize your own video

# TASK 1 FUNCTIONS

def opencv_lk2(prvs,next):
    hsv = np.zeros_like(prvs)
    hsv[...,1] = 255
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',bgr)
    cv2.waitKey(0)

def optical_flow_visualization(u, v, dil_size=0):

    H = u.shape[0]
    W = u.shape[1]
    hsv = np.zeros((H, W, 3))

    # u[np.isnan(u)]=0
    # v[np.isnan(v)]=

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(u, v)

    # mag[np.isnan(mag)]=0
    # ang[np.isnan(ang)]=0
    # mag[np.isinf(mag)]=0

    hsv[:,:,0] = ang * 180 / np.pi / 2
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # hsv[:,:,2] = ((mag - np.min(mag))/np.max(mag))*255
    hsv[:,:,1] = 255

    hsv[:, :, 0] = ndimage.grey_dilation(hsv[:, :, 0], size=(dil_size, dil_size))
    hsv[:, :, 2] = ndimage.grey_dilation(hsv[:, :, 2], size=(dil_size, dil_size))

    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.uint8)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # plt.imshow(rgb_flow)
    # plt.show()
    return rgb_flow

def flow_read(filename):
    # loads flow field F from png file

    # Read the image
    # -1 because the values are not limited to 255
    # OpenCV reads in BGR order of channels
    I = cv2.imread(filename, -1)

    # Representation:
    #   Vector of flow (u,v)
    #   Boolean if pixel has a valid value (is not an occlusion)
    F_u = (I[:, :, 2] - 2. ** 15) / 64
    F_v = (I[:, :, 1] - 2. ** 15) / 64
    F_valid = I[:, :, 0]

    # Matrix with vector (u,v) in the channel 0 and 1 and boolean valid in channel 2
    return np.transpose(np.array([F_u, F_v, F_valid]), axes=[1, 2, 0])

if __name__ == "__main__":
    main()
