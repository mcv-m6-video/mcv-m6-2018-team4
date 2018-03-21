import cv2
import numpy as np
import time
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from block_matching import block_matching
import time
from flow_utils_w4 import flow_visualization

def video_stabilization(sequence, GT = None):
    b_sz = (12,12)
    prev = sequence[0]

    H, W, C = prev.shape
    N = len(sequence)

    seq_stabilized = np.zeros((H,W,C,N))
    seq_stabilized[:,:,:,0] = prev
    GT_stabilized = []

    if GT != None:
        GT_stabilized = np.zeros((H,W,C,N))
        GT_stabilized[:,:,:,0] = GT[0]

    for i in range(1,N):
        t = time.time()
        next = sequence[i]
        flow = block_matching(prev, next, block_size=b_sz, step=b_sz, area=(20,20))

        mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
        uniques, counts = np.unique(mag, return_counts=True)
        mc_mag = uniques[counts.argmax()]

        uniques, counts = np.unique(ang, return_counts=True)
        mc_ang = uniques[counts.argmax()]

        u, v = pol2cart(mc_mag, mc_ang)

        affine_H = np.float32([[1, 0, -v],[0,1,-u]])

        next_stabilized = cv2.warpAffine(next,affine_H,(next.shape[1],next.shape[0]))

        if GT != None:
            frame_stabilized = cv2.warpAffine(GT[i],affine_H,(next.shape[1],next.shape[0]))
            GT_stabilized[:,:,:,i] = frame_stabilized
        prev = next_stabilized

        seq_stabilized[:,:,:,i] = next_stabilized

        print "Frame {} ({} sec)".format(i,time.time()-t)

    return seq_stabilized, GT_stabilized

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)