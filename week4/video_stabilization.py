import cv2
import numpy as np
import time
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from block_matching import block_matching

def video_stabilization(sequence):
    b_sz = (8,8)
    prev = sequence[0]

    H, W, C = prev.shape
    N = len(sequence)

    seq_stabilized = np.zeros((H,W,C,N))
    seq_stabilized[:,:,:,0] = prev

    for i in range(1,N):
        next = sequence[i]
        flow = block_matching(prev, next, block_size=b_sz*3, step=b_sz, area=b_sz)

        # mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])

        # mag = np.median(mag)
        # ang = np.median(ang)
        #
        # u, v = cv2.polarToCart(mag, ang)


        u = np.median(flow[:,:,0])
        v = np.median(flow[:,:,1])

        affine_H = np.float32([[1, 0, -u],[0,1,-v]])

        next_stabilized = cv2.warpAffine(next,affine_H,(next.shape[1],next.shape[0]))
        prev = next_stabilized

        seq_stabilized[:,:,:,i] = next_stabilized

    return seq_stabilized