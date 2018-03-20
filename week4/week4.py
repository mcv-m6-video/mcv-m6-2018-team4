import sys

from week4.lucas_kanade import optical_flow_lk

from week4.horn_schunk import optical_flow_hs

from week4.video_stabilization import video_stabilization

sys.path.append('..')
from week4.flow_utils_w4 import flow_read, flow_error, flow_visualization
import cv2
import os
from block_matching import block_matching
from dataset import Dataset
import matplotlib.pyplot as plt


def main():
    # TASK 1 - OPTICAL FLOW

    path = '../../Datasets/data_stereo_flow/training/'
    nsequence = '000045'
    # nsequence = '000157'

    F_gt = flow_read(os.path.join(path, 'flow_noc', nsequence + '_10.png'))

    sequence = []
    frame = cv2.imread(os.path.join(os.path.join(path, 'image_0', nsequence + '_10.png')))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sequence.append(frame)
    frame = cv2.imread(os.path.join(os.path.join(path, 'image_0', nsequence + '_11.png')))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sequence.append(frame)

    # TASK 1.1 - Block Matching
    flow = block_matching(sequence[0], sequence[1], block_size=(3 * 8, 3 * 8), step=(8, 8), area=(8, 8))

    flow_error(F_gt, flow)
    rgb_flow = flow_visualization(flow[:, :, 0], flow[:, :, 1], 0)

    plt.imshow(rgb_flow)
    plt.show()

    # TASK 1.2 - Block Matching vs Other Techniques
    # sigma = 15;
    # u, v = optical_flow_lk(sequence[0],sequence[1],sigma)

    # alpha = 0.5;
    # u, v = optical_flow_hs(sequence[0],sequence[1],alpha)

    # rgb_flow = flow_visualization(u,v)

    # flow = cv2.calcOpticalFlowFarneback(sequence[0], sequence[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # TASK 2 - VIDEO STABILIZATION

    # Read traffic dataset
    traffic_dataset = Dataset('traffic', 951, 1050)

    traffic = traffic_dataset.readInput()
    traffic_GT = traffic_dataset.readGT()

    # Split dataset
    # traffic_train = traffic[:len(traffic)/2]
    # traffic_test = traffic[len(traffic)/2:]
    # traffic_test_GT = traffic_GT[len(traffic)/2:]

    # TASK 2.1 - Video stabilization with Block Matching
    # video_stabilization(traffic)

    # TASK 2.2 - Block Matching Stabilization vs Other Techniques

    # TASK 2.3 - Stabilize your own video


if __name__ == "__main__":
    main()
