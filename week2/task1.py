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
    # dataset = Dataset('highway',1050, 1350)
    # dataset = Dataset('fall', 1461, 1560)
    dataset = Dataset('traffic', 951, 1050)

    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    # Split dataset
    train = imgs[:len(imgs)/2]
    test = imgs[len(imgs)/2:]
    test_GT = imgs_GT[len(imgs)/2:]

    # # Background substraction
    # g = GaussianModelling()
    # g.fit(train)
    # results = g.predict(test)
    #
    # # Evaluation sklearn
    # t = time.time()
    # metrics = ev.getMetrics(test_GT,results)
    # elapsed = time.time() - t
    # sys.stdout.write(str(elapsed) + ' sec \n')

    grid_search(train, test, test_GT)

    # f1score_alpha(train, test, test_GT)

def f1score_alpha(train, test, test_GT):
    # Task 1.2 - F1-score vs Alpha
    alpha_range = np.arange(0,3,0.1)

    metrics_array = []

    for alpha in alpha_range:
        sys.stdout.write("(alpha=" + str(alpha)+") ")

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
    x = alpha_range
    metrics_array = np.array(metrics_array)
    y = metrics_array[:, 2]
    axis = ["Alpha", "F1-score"]
    labels = ["traffic"]
    ev.plotGraphics(x, y, axis, labels)

def grid_search(train, test, test_GT):
    # Task 1.3 - F1-score vs Alpha
    alpha_range = np.arange(0, 3, 0.1)
    ro_range = np.arange(0, 1, 0.03)

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
    axis = ["Ro","Alpha", "F1-score"]

    X, Y = np.meshgrid(ro_range,alpha_range)
    Z = f1_matrix

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')

    ax.set_xlabel(axis[0])
    ax.set_ylabel(axis[1])
    ax.set_zlabel(axis[2])

    plt.savefig('grid_search.png',dpi=300)
    plt.show()



if __name__ == "__main__":
    main()
