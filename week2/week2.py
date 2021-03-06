import sys
sys.path.append('..')
import cv2
import evaluation as ev
import numpy as np
import os
import time

from dataset import Dataset
from gaussian_modelling import GaussianModelling

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():

    # Read dataset
    # dataset = Dataset('highway',1051, 1350)
    # dataset = Dataset('fall', 1461, 1560)
    dataset = Dataset('traffic', 951, 1050)

    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    # Split dataset
    train = imgs[:len(imgs)/2]
    test = imgs[len(imgs)/2:]
    test_GT = imgs_GT[len(imgs)/2:]

    # TASK 1.1
    # single_execution(train, test, test_GT, 1.8, 0.15)

    # TASK 1.2
    # f1score_alpha(train, test, test_GT)

    # TASK 1.3
    # precision_recall_curve(train, test, test_GT)

    # TASK 2
    # grid_search(train, test, test_GT)

def single_execution(train, test, test_GT, alpha, ro):

    sys.stdout.write('Computing background substraction... ')
    # Background substraction
    g = GaussianModelling(alpha=alpha,adaptive_ratio=ro)
    g.fit(train)
    results = g.predict(test)

    # Evaluation sklearn
    t = time.time()
    metrics = ev.getMetrics(test_GT,results)
    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')

    print "Recall: " + str(metrics[0] * 100)
    print "Precision: " + str(metrics[1] * 100)
    print "F1: " + str(metrics[2] * 100)

def f1score_alpha(train, test, test_GT):
    # Task 1.2 - F1-score vs Alpha
    alpha_range = np.around(np.arange(1.5,2.5,0.1),decimals=2)

    metrics_array = []

    for alpha in alpha_range:
        sys.stdout.write("(alpha=" + str(np.around(alpha,decimals=2))+") ")

        # Background substraction
        g = GaussianModelling(alpha=alpha)
        g.fit(train)
        results = g.predict(test)

        # Evaluation sklearn
        t = time.time()
        metrics = ev.getMetrics(test_GT, results)
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

        metrics_array.append(metrics);

    # TASK 2.2 - Plot F1-score vs Alpha
    x = [alpha_range, alpha_range, alpha_range]
    metrics_array = np.array(metrics_array)
    y = [metrics_array[:, 0], metrics_array[:, 1],metrics_array[:, 2]]

    f1_max = np.max(metrics_array[:, 2])
    f1_max_idx = np.argmax(metrics_array[:, 2])
    best_alpha = alpha_range[f1_max_idx]
    print "F1: " + str(np.around(f1_max,decimals=4)) + " (alpha="+str(best_alpha)+")"

    axis = ["Alpha", "F1-score"]
    labels = ["Precision", "Recall", "F1"]
    ev.plotGraphics(x, y, axis, labels)

def precision_recall_curve(train, test, test_GT):

    sys.stdout.write('Computing Precision-Recall curve... ')
    # Background substraction
    g = GaussianModelling(adaptive_ratio=0.15,grayscale_modelling=False)
    g.fit(train)
    scores = g.predict_probabilities(test)

    t = time.time()
    precision, recall, auc_val = ev.getPR_AUC(test_GT, scores)
    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')

    plt.step(recall, precision, color='g', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='g')
    print "AUC: "+ str(auc_val)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    # plt.title("Precision-Recall curve (AUC=" + str(auc_val) + ")" )
    plt.title("Precision-Recall curve - Fall" )
    plt.show()

def grid_search(train, test, test_GT):
    alpha_range = np.around(np.arange(2.5, 4.52, 0.1),decimals=2)
    ro_range = np.around(np.arange(0, 0.4, 0.05),decimals=2)

    f1_matrix = np.zeros([len(alpha_range),len(ro_range)])

    for i in range(len(alpha_range)):
        alpha = alpha_range[i]
        for j in range(len(ro_range)):
            ro = ro_range[j]
            sys.stdout.write("(alpha=" + str(alpha)+", ro=" + str(ro)+") ")

            # Background substraction
            g = GaussianModelling(alpha=alpha,adaptive_ratio=ro)
            g.fit(train)
            results = g.predict(test)

            # Evaluation sklearn
            t = time.time()
            metrics = ev.getMetrics(test_GT, results)
            elapsed = time.time() - t
            sys.stdout.write(str(elapsed) + ' sec \n')

            f1_matrix[i,j] = metrics[2]

    # Plot grid search
    X, Y = np.meshgrid(ro_range,alpha_range)
    Z = f1_matrix

    f1_max = np.max(f1_matrix)
    f1_max_idx = np.argmax(f1_matrix)
    best_alpha = Y.flatten()[f1_max_idx]
    best_ro = X.flatten()[f1_max_idx]

    print "F1: " + str(np.around(f1_max, decimals=5)) + " (alpha=" + str(best_alpha) + ", ro=" + str(best_ro)+")"

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    axis = ["Ro","Alpha", "F1-score"]
    ax.set_xlabel(axis[0])
    ax.set_ylabel(axis[1])
    ax.set_zlabel(axis[2])
    # plt.savefig('grid_search.png',dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
