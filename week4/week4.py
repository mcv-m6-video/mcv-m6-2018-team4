import sys

from lucas_kanade import optical_flow_lk
import time

from horn_schunk import optical_flow_hs

from video_stabilization import video_stabilization

sys.path.append('..')
sys.path.append('../week3/')
sys.path.append('../week2/')
from flow_utils_w4 import flow_read, flow_error, flow_visualization
import cv2
import os
from block_matching import block_matching
from dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.metrics import auc

from week3 import *
from gaussian_modelling import GaussianModelling

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
    flow = block_matching(sequence[0], sequence[1], block_size=(16, 16), step=(8, 8), area=(24, 24), area_step=(1,1), error_thresh=1, verbose=True)

    flow_error(F_gt, flow)
    rgb_flow = flow_visualization(flow[:, :, 0], flow[:, :, 1], 0)
    plt.imshow(rgb_flow)
    plt.show()

    # grid_search_block_matching(sequence, F_gt)

    # TASK 1.2 - Block Matching vs Other Techniques
    # sigma = 120;
    # u, v = optical_flow_lk(sequence[0],sequence[1],sigma)
    #
    # alpha = 0.5;
    # u, v = optical_flow_hs(sequence[0],sequence[1],alpha)


    flow = cv2.calcOpticalFlowFarneback(sequence[0], sequence[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    rgb_flow = flow_visualization(flow[:,:,0],flow[:,:,1],3)
    # rgb_flow = flow_visualization(F_gt[:,:,0],F_gt[:,:,1],3)
    plt.imshow(rgb_flow)
    plt.show()

    # area = np.load('grid_search/area_size_vector5.npy')
    # block = np.load('grid_search/block_size_vector5.npy')
    # mse = np.load('grid_search/mse_results5.npy')
    # pepn = np.load('grid_search/pepn_results5.npy')
    # plot_grid_search(mse, area, block, ['Area', 'Block size', 'MSEN'])
    # # plot_grid_search(pepn, area, block, ['Area', 'Block size', 'PEPN'])

    # TASK 2 - VIDEO STABILIZATION

    # Read traffic dataset
    traffic_dataset = Dataset('traffic', 951, 1050)

    traffic = traffic_dataset.readInput()
    traffic_GT = traffic_dataset.readGT()

    # Split dataset
    train = traffic[:len(traffic)/2]
    test = traffic[len(traffic)/2:]
    test_GT = traffic_GT[len(traffic)/2:]

    # TASK 2.1 - Video stabilization with Block Matching
    seq_stab, GT_stab = video_stabilization(traffic,traffic_GT)
    np.save('seq_stab.npy',seq_stab)
    np.save('GTstab.npy',GT_stab)


    # seq_stab = np.load('seq_stab.npy')
    # GT_stab = np.load('GT_stab.npy')

    traffic_np = np.zeros((traffic[0].shape[0],traffic[0].shape[1],3,len(traffic)))
    for i in range(len(traffic)):
        traffic_np[:,:,:,i] = traffic[i]

    make_gif(traffic_np,'traffic.gif')
    make_gif(GT_stab,'GT_stab.gif')


    # TASK 2.2 - Block Matching Stabilization vs Other Techniques

    seq_stab_np = np.load('video_stabilization/seq_stab.npy')
    GT_stab_np = np.load('video_stabilization/GTstab.npy')

    valid_frames_np = get_valid_mask(seq_stab_np)

    seq_stab = []
    GT_stab = []
    valid_frames = []
    for i in range(seq_stab_np.shape[3]):
        seq_stab.append(seq_stab_np[:,:,:,i].astype(np.uint8))
        GT_stab.append(GT_stab_np[:,:,:,i].astype(np.uint8))
        valid_frames.append(valid_frames_np[:,:,i])

    # Split dataset
    train_stab = seq_stab[:len(seq_stab)/2]
    test_stab = seq_stab[len(seq_stab)/2:]
    valid_frames_test = valid_frames[len(seq_stab)/2:]
    test_GT_stab = GT_stab[len(seq_stab)/2:]


    seq_stab2_np = np.load('video_stabilization/ORB_sequence.npy')
    GT_stab2_np = np.load('video_stabilization/ORB_sequence_GT.npy')
    valid_frames2_np = get_valid_mask2(seq_stab2_np)

    seq_stab2 = []
    GT_stab2 = []
    valid_frames2 = []
    for i in range(seq_stab2_np.shape[0]):
        seq_stab2.append(seq_stab2_np[i,0,:,:,:].astype(np.uint8))
        GT_stab2.append(GT_stab2_np[i,0,:,:,:].astype(np.uint8))
        valid_frames2.append(valid_frames2_np[:,:,i])

    # Split dataset
    train_stab2 = seq_stab2[:len(seq_stab2)/2]
    test_stab2 = seq_stab2[len(seq_stab2)/2:]
    test_GT_stab2 = GT_stab2[len(seq_stab2)/2:]
    valid_frames_test2 = valid_frames2[len(seq_stab)/2:]

    # precision_recall_curve(train, train_stab, train_stab2, test, test_stab, test_stab2, test_GT, test_GT_stab, test_GT_stab2, 0.15, 4, 330, valid_frames_test, valid_frames_test2, True)
    # results, metrics = task2_pipeline(train_stab, test_stab, test_GT_stab, 3.5, 0.15, 4, 330, False)
    # results, metrics = task3_morphology_traffic(train_stab2, test_stab2, test_GT_stab2, 3.5, 0.15, 4, 330, False,valid_frames2)
    results, metrics = task3_morphology_traffic(train_stab, test_stab, test_GT_stab, 3.5, 0.15, 4, 330, False, valid_frames)
    # results, metrics = task3_morphology_traffic(train, test, test_GT, 3.5, 0.15, 4, 330, False, valid_frames)

    make_gif_w3(results)

    # TASK 2.3 - Stabilize your own video

    cap = cv2.VideoCapture('our_sequence.mp4')

    our_sequence = []
    ret = True
    while (ret):
        ret, frame = cap.read()
        our_sequence.append(frame)

    cap.release()

    our_sequence, GT_stab = video_stabilization(our_sequence)
    our_sequence = our_sequence.astype(np.uint8)
    np.save('our_sequence.npy',our_sequence)
    # seq_stab = np.load('seq_stab.npy')
    make_gif(our_sequence,'our_sequence.gif')

def get_valid_mask(sequence):

    valid_frames = np.zeros((sequence.shape[0],sequence.shape[1],sequence.shape[3]),dtype=np.bool)
    for i in range(sequence.shape[3]):
        frame = sequence[:,:,:,i]
        frame = (frame == 0)
        valid_frame = np.logical_and(frame[:, :, 0], frame[:, :, 1], frame[:, :, 2])
        valid_frames[:,:,i] = valid_frame

    return valid_frames

def get_valid_mask2(sequence):

    valid_frames = np.zeros((sequence.shape[2],sequence.shape[3],sequence.shape[0]),dtype=np.bool)
    for i in range(sequence.shape[0]):
        frame = sequence[i,:,:,:]
        frame = (frame == 0)
        valid_frame = np.logical_and(frame[0, :, :, 0], frame[0, :, :, 1], frame[0, :, :, 2])
        valid_frames[:,:,i] = valid_frame

    return valid_frames

def make_gif_w3(results):
    ims = []
    fig=plt.figure()
    for i in range(len(results)):
        im = plt.imshow(results[i],cmap='gray',animated=True)
        plt.axis("off")
        ims.append([im])

    anim = animation.ArtistAnimation(fig, ims,interval=len(results), blit=True)
    anim.save('animation.gif', writer='imagemagick', fps=10)
    plt.show()

    return

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
    plot_grid_search(mse_results,area_mult_vector,block_size_vector,['Area multiplier','Block size', 'MSE'], save_to_file='MSE.png')
    plot_grid_search(pepn_results,area_mult_vector,block_size_vector,['Area multiplier','Block size', 'PEPN'], save_to_file='PEPN.png')


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

def precision_recall_curve(train, train_stab, train_stab2, test, test_stab, test_stab2, test_GT, test_GT_stab, test_GT_stab2, ro, conn, p, valid_frames_test,valid_frames_test2, prints=True):
    tt = time.time()
    sys.stdout.write('Computing Precision-Recall curve... ')

    alpha_range = np.around(np.arange(0, 20.2, 0.5), decimals=2)

    metrics_array = []
    metrics2_array = []
    metrics_old_array = []

    for alpha in alpha_range:
        if prints:
            t = time.time()
            sys.stdout.write("(alpha=" + str(np.around(alpha, decimals=2)) + ") ")

        # results_old, metrics_old = task2_pipeline(train, test, test_GT, alpha, ro, conn, p, False)
        results, metrics_old = task3_morphology_traffic(train, test, test_GT, alpha, ro, conn, p, False)
        #
        # results, metrics = task2_pipeline(train_stab, test_stab, test_GT_stab, alpha, ro, conn, p, False)
        results, metrics = task3_morphology_traffic(train_stab, test_stab, test_GT_stab, alpha, ro, conn, p, False, valid_frames_test)
        #
        # results2, metrics2 = task2_pipeline(train_stab2, test_stab2, test_GT_stab2, alpha, ro, conn, p, False)
        # results, metrics2 = task3_morphology_traffic(train_stab2, test_stab2, test_GT_stab2, alpha, ro, conn, p, False, valid_frames_test2)

        metrics_array.append(metrics)
        # metrics2_array.append(metrics2)
        metrics_old_array.append(metrics_old)

        if prints:
            elapsed = time.time() - t
            sys.stdout.write(str(elapsed) + ' sec \n')

    precision = np.array(metrics_array)[:, 0]
    recall = np.array(metrics_array)[:, 1]
    auc_val = auc(recall, precision)

    precision_old = np.array(metrics_old_array)[:, 0]
    recall_old = np.array(metrics_old_array)[:, 1]
    auc_val_old = auc(recall_old, precision_old)

    # precision2 = np.array(metrics2_array)[:, 0]
    # recall2 = np.array(metrics2_array)[:, 1]
    # auc_val2 = auc(recall2, precision2)

    sys.stdout.write("(auc=" + str(np.around(auc_val, decimals=4)) + ") ")
    sys.stdout.write("(auc_old=" + str(np.around(auc_val_old, decimals=4)) + ") ")
    # sys.stdout.write("(auc2=" + str(np.around(auc_val2, decimals=4)) + ") ")

    elapsed = time.time() - tt
    sys.stdout.write(str(elapsed) + ' sec \n')

    np.save('metrics_week4_pr.npy',metrics_array)
    # np.save('metrics_week4_2_pr.npy',metrics2_array)
    np.save('metrics_week3_pr.npy',metrics_old_array)

    if prints:
        # plt.plot(recall2, precision2, color='b', label='ORB stabilization')
        plt.plot(recall, precision, color='g', label='With stabilization')
        plt.plot(recall_old, precision_old, color='r', label='No stabilization')
        print "AUC stabilized: "+ str(auc_val)
        print "AUC: "+ str(auc_val_old)
        # print "AUC2: "+ str(auc_val2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title("Precision-Recall curves")
        plt.legend()
        plt.show()

    return auc_val

def make_gif(results, gifname):

    H, W, C, N = results.shape

    ims = []
    fig = plt.figure()
    for i in range(N):
        im = plt.imshow(cv2.cvtColor(results[:,:,:,i].astype(np.uint8), cv2.COLOR_BGR2RGB),animated=True)
        # im = plt.imshow(results[:, :, :, i], animated=True)
        plt.axis("off")
        ims.append([im])

    anim = animation.ArtistAnimation(fig, ims, interval=len(results), blit=True)
    anim.save(gifname, writer='imagemagick', fps=20)
    plt.show()

    return

if __name__ == "__main__":
    main()
