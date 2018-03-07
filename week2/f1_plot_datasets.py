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
    datasets = [Dataset('highway',1050, 1350), Dataset('fall', 1461, 1560), Dataset('traffic', 951, 1050)]
    metricses = []
    for dataset in range(3):
        print "Computing dataset " + str(dataset+1)
        imgs = datasets[dataset].readInput()
        imgs_GT = datasets[dataset].readGT()

        # Split dataset
        train = imgs[:len(imgs)/2]
        test = imgs[len(imgs)/2:]
        test_GT = imgs_GT[len(imgs)/2:]

        metrics_array = f1score_alpha(train, test, test_GT)
        metricses.append(metrics_array)

    # TASK 2.2 - Plot F1-score vs Alpha

    alpha_range = np.around(np.arange(0,6,0.1),decimals=2)

    metricses = np.array(metricses)

    x = [alpha_range, alpha_range, alpha_range]
    y = [metricses[0, :], metricses[1, :],metricses[2, :]]

    axis = ["Alpha", "F1-score"]
    labels = ["Highway", "Fall", "Traffic"]
    ev.plotGraphics(x, y, axis, labels)


def f1score_alpha(train, test, test_GT):
    # Task 1.2 - F1-score vs Alpha
    alpha_range = np.around(np.arange(0,6,0.1),decimals=2)

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

    metrics_array= np.array(metrics_array)
    return metrics_array[:,2]
if __name__ == "__main__":
    main()
