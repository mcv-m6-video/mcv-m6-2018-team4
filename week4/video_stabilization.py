import cv2
import numpy as np
import time
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from block_matching import block_matching

def video_stabilization(sequence):
    b_sz = (8,8)
    prev = sequence[:,:,0]

    seq_stabilized = np.zeros_like(sequence)
    seq_stabilized[:,:,0] = prev

    for i in range(1,sequence.shape[2]):
        next = sequence[:,:,i]
        flow = block_matching(prev, next, block_size=b_sz*3, step=b_sz, area=b_sz)

        u = np.median(flow[:,:,0])
        v = np.median(flow[:,:,1])

        affine_H = np.float32([[1, 0, u],[0,1,v]])

        next_stabilized = cv2.warpAffine(next,affine_H,next.shape)

        prev = next
