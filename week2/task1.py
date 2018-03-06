import cv2
import evaluation as ev
import numpy as np
import os
import time
from dataset import Dataset
from gaussian_modelling import GaussianModelling
import sys


def main():

    # dataset = Dataset('highway',1050, 1350)
    # dataset = Dataset('fall', 1461, 1560)
    dataset = Dataset('traffic', 951, 1050)

    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    train = imgs[:len(imgs)/2]
    test = imgs[len(imgs)/2:]
    test_GT = imgs_GT[len(imgs)/2:]

    # Background substraction
    g = GaussianModelling()
    g.fit(train)
    results = g.predict(test)

    # Evaluation
    t = time.time()
    conf_matrix = ev.ConfusionMatrix(test_GT, results, False)
    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')
    metrics = ev.Metrics(conf_matrix, True)


    # Task 2.2 - F1-score vs Alpha
    alpha_range = np.arange(0,20,0.1)
    metrics_array = [];
    for alpha in alpha_range:
        sys.stdout.write("(alpha=" + str(alpha)+") ")
        # sys.stdout.flush()
        # Background substraction
        g = GaussianModelling(alpha=alpha)
        g.fit(train)
        results = g.predict(test)

        # Evaluation
        t = time.time()
        conf_matrix = ev.ConfusionMatrix(test_GT, results, False)
        metrics = ev.Metrics(conf_matrix, False)
        metrics_array.append(metrics);
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

    # TASK 2.2 - Plot F1-score vs Alpha
    x = alpha_range
    metrics_array = np.array(metrics_array)
    y = metrics_array[:, 3]
    axis = ["Alpha", "F1-score"]
    labels = ["traffic"]
    ev.plotGraphics(x, y, axis, labels)



if __name__ == "__main__":
    main()
