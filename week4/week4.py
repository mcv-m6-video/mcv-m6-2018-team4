import sys

from lucas_kanade import optical_flow_lk
import time

from horn_schunk import optical_flow_hs

from video_stabilization import video_stabilization

sys.path.append('..')
from flow_utils_w4 import flow_read, flow_error, flow_visualization
import cv2
import os
from block_matching import block_matching
from dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np


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
    # flow = block_matching(sequence[0], sequence[1], block_size=(16, 16), step=(8, 8), area=(24, 24), area_step=(1,1), error_thresh=1, verbose=True)
    #
    # flow_error(F_gt, flow)
    # rgb_flow = flow_visualization(flow[:, :, 0], flow[:, :, 1], 0)
    #
    # plt.imshow(rgb_flow)
    # plt.show()
    # grid_search_block_matching(sequence, F_gt)

    # TASK 1.2 - Block Matching vs Other Techniques
    # sigma = 15;
    # u, v = optical_flow_lk(sequence[0],sequence[1],sigma)

    # alpha = 0.5;
    # u, v = optical_flow_hs(sequence[0],sequence[1],alpha)

    # rgb_flow = flow_visualization(u,v)

    # flow = cv2.calcOpticalFlowFarneback(sequence[0], sequence[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # TASK 2 - VIDEO STABILIZATION

    # Read traffic dataset
    # traffic_dataset = Dataset('traffic', 951, 1050)

    # traffic = traffic_dataset.readInput()
    # traffic_GT = traffic_dataset.readGT()

    # Split dataset
    # traffic_train = traffic[:len(traffic)/2]
    # traffic_test = traffic[len(traffic)/2:]
    # traffic_test_GT = traffic_GT[len(traffic)/2:]

    # TASK 2.1 - Video stabilization with Block Matching
    # video_stabilization(traffic)

    # TASK 2.2 - Block Matching Stabilization vs Other Techniques

    # TASK 2.3 - Stabilize your own video


def grid_search_block_matching(sequence, F_gt):
    block_size_vector = np.arange(4, 21, step=2)
    area_mult_vector = np.arange(1, 4, step=1)
    mse_results = np.zeros([len(block_size_vector), len(area_mult_vector)])
    pepn_results = np.zeros([len(block_size_vector), len(area_mult_vector)])

    print('Grid size: {}*{}={}'.format(len(block_size_vector), len(area_mult_vector),
                                       len(block_size_vector) * len(area_mult_vector)))
    i = 0
    for bs in block_size_vector:
        j = 0
        for am in area_mult_vector:
            start = time.time()
            F_est = block_matching(sequence[0], sequence[1], block_size=(bs, bs), step=(bs, bs),
                                   area=(am * bs, am * bs))
            MSE, PEPN = flow_error(F_gt, F_est)
            print('Block Size: {} Area Multiplier: {} -> MSE: {} PEPN: {} Time: {}sec'.format(bs, am, MSE, PEPN,
                                                                                              time.time() - start))
            mse_results[i, j] = MSE
            pepn_results[i, j] = PEPN
            j += 1
        i += 1
    np.save('mse_results', mse_results)
    np.save('pepn_results', pepn_results)
    np.save('block_size_vector', block_size_vector)
    np.save('area_mult_vector', area_mult_vector)
    plot_grid_search(mse_results,block_size_vector,area_mult_vector,['Block size', 'Area multiplier', 'MSE'], save_to_file='MSE.png')


def plot_grid_search(values, x_range, y_range, legend, save_to_file=None):
    # Plot grid search
    X, Y = np.meshgrid(x_range, y_range)
    Z = values

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.set_xlabel(legend[0])
    ax.set_ylabel(legend[1])
    ax.set_zlabel(legend[2])
    if save_to_file is not None:
        plt.savefig(save_to_file, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
