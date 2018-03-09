import sys
sys.path.append('..')
sys.path.append('../week2/')
import cv2
import evaluation as ev
import numpy as np
from numpy.matlib import repmat
import os
import time

from dataset import Dataset
from gaussian_modelling import GaussianModelling

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import binary_fill_holes, generate_binary_structure, binary_dilation


def main():
    # Week 2 Best configurations
    #   - Highway   (a=2.4,ro=0.15) F1=0.6319 - conn4 (65.3461132183114)
    #   - Fall      (a=3.2,ro=0.05) F1=0.6896 - conn4 (73.94992293059627)
    #   - Traffic   (a=3.5,ro=0.15) F1=0.6376 - conn8 (63.954441547032424) conn4(63.91668494564192)

    # dataset_name = 'highway'
    # dataset_name = 'fall'
    dataset_name = 'traffic'

    if dataset_name == 'highway':
        frames_range = (1051, 1350)
        alpha = 2.4
        ro = 0.15

    elif dataset_name == 'fall':
        frames_range = (1461, 1560)
        alpha = 3.2
        ro = 0.05

    elif dataset_name == 'traffic':
        frames_range = ( 951, 1050)
        alpha = 3.5
        ro = 0.15

    # Read dataset
    dataset = Dataset(dataset_name,frames_range[0], frames_range[1])

    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    # Split dataset
    train = imgs[:len(imgs)/2]
    test = imgs[len(imgs)/2:]
    test_GT = imgs_GT[len(imgs)/2:]

    # TASK 1 - Hole Filling
    task1_hole_filling(train, test, test_GT, alpha, ro, 4)

    # TASK 2 - Area Filtering

    # TASK 3 - Morphology

    # TASK 4 - Shadow removal

    # TASK 5 - Show improvements (PR-curve/AUC)

def task1_hole_filling(train, test, test_GT, alpha, ro, conn):

    if conn == 4:
        el = generate_binary_structure(2,1)
    elif conn == 8:
        el = generate_binary_structure(2,2)
    else:
        print "Connectivity not valid"
        return

    results = background_substraction(train, test, alpha, ro)

    t = time.time()
    sys.stdout.write('Computing hole filling... ')

    for image in range(len(results)):
        results[image,:,:] = binary_fill_holes(results[image,:,:],el)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')

    results_evaluation(results, test_GT)

    return results

def background_substraction(train, test, alpha, ro):

    # Background substraction
    t = time.time()
    sys.stdout.write('Computing background substraction... ')
    g = GaussianModelling(alpha=alpha,adaptive_ratio=ro)
    g.fit(train)
    results = g.predict(test)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')

    return results

def results_evaluation(results, test_GT):
    # Evaluation sklearn
    sys.stdout.write('Evaluating results... ')
    t = time.time()
    metrics = ev.getMetrics(test_GT, results)
    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n\n')

    print "Recall: " + str(metrics[0] * 100)
    print "Precision: " + str(metrics[1] * 100)
    print "F1: " + str(metrics[2] * 100)

    return metrics

def task1_hole_filling_nofor(train, test, test_GT, alpha, ro, conn):

    if conn == 4:
        el = generate_binary_structure(2,1)
    elif conn == 8:
        el = generate_binary_structure(2,2)
    else:
        print "Connectivity not valid"
        return

    results = background_substraction(train, test, alpha, ro)

    el3d = np.repeat(el[:,:,np.newaxis],len(results),axis=2)

    t = time.time()
    sys.stdout.write('Computing hole filling... ')

    results = binary_fill_holes(results)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')

    results_evaluation(results, test_GT)

# def single_execution_wek2(train, test, test_GT, alpha, ro):
#     sys.stdout.write('Computing background substraction... ')
#     # Background substraction
#     g = GaussianModelling(alpha=alpha, adaptive_ratio=ro)
#     g.fit(train)
#     results = g.predict(test)
#
#     # Evaluation sklearn
#     t = time.time()
#     metrics = ev.getMetrics(test_GT, results)
#     elapsed = time.time() - t
#     sys.stdout.write(str(elapsed) + ' sec \n')
#
#     print "Recall: " + str(metrics[0] * 100)
#     print "Precision: " + str(metrics[1] * 100)
#     print "F1: " + str(metrics[2] * 100)
#
# def f1score_alpha(train, test, test_GT):
#     # Task 1.2 - F1-score vs Alpha
#     alpha_range = np.around(np.arange(1.5,2.5,0.1),decimals=2)
#
#     metrics_array = []
#
#     for alpha in alpha_range:
#         sys.stdout.write("(alpha=" + str(np.around(alpha,decimals=2))+") ")
#
#         # Background substraction
#         g = GaussianModelling(alpha=alpha)
#         g.fit(train)
#         results = g.predict(test)
#
#         # Evaluation sklearn
#         t = time.time()
#         metrics = ev.getMetrics(test_GT, results)
#         elapsed = time.time() - t
#         sys.stdout.write(str(elapsed) + ' sec \n')
#
#         metrics_array.append(metrics);
#
#     # TASK 2.2 - Plot F1-score vs Alpha
#     x = [alpha_range, alpha_range, alpha_range]
#     metrics_array = np.array(metrics_array)
#     y = [metrics_array[:, 0], metrics_array[:, 1],metrics_array[:, 2]]
#
#     f1_max = np.max(metrics_array[:, 2])
#     f1_max_idx = np.argmax(metrics_array[:, 2])
#     best_alpha = alpha_range[f1_max_idx]
#     print "F1: " + str(np.around(f1_max,decimals=4)) + " (alpha="+str(best_alpha)+")"
#
#     axis = ["Alpha", "F1-score"]
#     labels = ["Precision", "Recall", "F1"]
#     ev.plotGraphics(x, y, axis, labels)
#
# def precision_recall_curve(train, test, test_GT):
#
#     sys.stdout.write('Computing Precision-Recall curve... ')
#     # Background substraction
#     g = GaussianModelling(adaptive_ratio=0.15,grayscale_modelling=False)
#     g.fit(train)
#     scores = g.predict_probabilities(test)
#
#     t = time.time()
#     precision, recall, auc_val = ev.getPR_AUC(test_GT, scores)
#     elapsed = time.time() - t
#     sys.stdout.write(str(elapsed) + ' sec \n')
#
#     plt.step(recall, precision, color='g', alpha=0.2, where='post')
#     plt.fill_between(recall, precision, step='post', alpha=0.2, color='g')
#     print "AUC: "+ str(auc_val)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.0])
#     plt.xlim([0.0, 1.0])
#     # plt.title("Precision-Recall curve (AUC=" + str(auc_val) + ")" )
#     plt.title("Precision-Recall curve - Fall" )
#     plt.show()
#
# def grid_search(train, test, test_GT):
#     alpha_range = np.around(np.arange(2.5, 4.52, 0.1),decimals=2)
#     ro_range = np.around(np.arange(0, 0.4, 0.05),decimals=2)
#
#     f1_matrix = np.zeros([len(alpha_range),len(ro_range)])
#
#     for i in range(len(alpha_range)):
#         alpha = alpha_range[i]
#         for j in range(len(ro_range)):
#             ro = ro_range[j]
#             sys.stdout.write("(alpha=" + str(alpha)+", ro=" + str(ro)+") ")
#
#             # Background substraction
#             g = GaussianModelling(alpha=alpha,adaptive_ratio=ro)
#             g.fit(train)
#             results = g.predict(test)
#
#             # Evaluation sklearn
#             t = time.time()
#             metrics = ev.getMetrics(test_GT, results)
#             elapsed = time.time() - t
#             sys.stdout.write(str(elapsed) + ' sec \n')
#
#             f1_matrix[i,j] = metrics[2]
#
#     # Plot grid search
#     X, Y = np.meshgrid(ro_range,alpha_range)
#     Z = f1_matrix
#
#     f1_max = np.max(f1_matrix)
#     f1_max_idx = np.argmax(f1_matrix)
#     best_alpha = Y.flatten()[f1_max_idx]
#     best_ro = X.flatten()[f1_max_idx]
#
#     print "F1: " + str(np.around(f1_max, decimals=5)) + " (alpha=" + str(best_alpha) + ", ro=" + str(best_ro)+")"
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='plasma')
#     axis = ["Ro","Alpha", "F1-score"]
#     ax.set_xlabel(axis[0])
#     ax.set_ylabel(axis[1])
#     ax.set_zlabel(axis[2])
#     # plt.savefig('grid_search.png',dpi=300)
#     plt.show()

if __name__ == "__main__":
    main()
