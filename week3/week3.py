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
from scipy.ndimage import binary_fill_holes, generate_binary_structure, label, labeled_comprehension


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

    else:
        print "Invalid dataset name"
        return

    # Read dataset
    dataset = Dataset(dataset_name,frames_range[0], frames_range[1])

    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    # Split dataset
    train = imgs[:len(imgs)/2]
    test = imgs[len(imgs)/2:]
    test_GT = imgs_GT[len(imgs)/2:]

    # TASK 1 - Hole Filling
    # task1_pipeline(train, test, test_GT, alpha, ro, 4, True)

    # TASK 2 - Area Filtering
    task2_pipeline(train, test, test_GT, alpha, ro, 4, 100, True)
    # f1_p(train, test, test_GT, alpha, ro, 4)

    # TASK 3 - Morphology

    # TASK 4 - Shadow removal

    # TASK 5 - Show improvements (PR-curve/AUC)

def task1_pipeline(train, test, test_GT, alpha, ro, conn, prints):

    results = background_substraction(train, test, alpha, ro)

    results = hole_filling(results,conn)

    results_evaluation(results, test_GT)

    return results

def task2_pipeline(train, test, test_GT, alpha, ro, conn, p, prints):

    results = background_substraction(train, test, alpha, ro, prints)
    results = hole_filling(results,conn, prints)
    results = area_filtering(results,conn, p, prints)
    metrics = results_evaluation(results, test_GT, prints)

    return results, metrics

def background_substraction(train, test, alpha, ro, prints):

    if prints:
        t = time.time()
        sys.stdout.write('Computing background substraction... ')
    g = GaussianModelling(alpha=alpha,adaptive_ratio=ro)
    g.fit(train)
    results = g.predict(test)

    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

    return results

def hole_filling(images, conn, prints):

    if conn == 4:
        el = generate_binary_structure(2,1)
    elif conn == 8:
        el = generate_binary_structure(2,2)
    else:
        print "Connectivity not valid"
        return

    if prints:
        t = time.time()
        sys.stdout.write('Computing hole filling... ')

    for image in range(len(images)):
        images[image, :, :] = binary_fill_holes(images[image, :, :], el)


    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

    return images

def area_filtering(images, conn, pixels, prints):

    if conn == 4:
        el = generate_binary_structure(2,1)
    elif conn == 8:
        el = generate_binary_structure(2,2)
    else:
        print "Connectivity not valid"
        return

    if prints:
        t = time.time()
        sys.stdout.write('Computing area filtering... ')

    for image in range(len(images)):
        image_labeled, nlbl = label(images[image, :, :], el)

        lbls = np.arange(1, nlbl + 1)

        if nlbl>0:
            info = labeled_comprehension(images, image_labeled, lbls, np.sum, float, 0)
            valid_lbls = lbls[info > pixels]

            result = np.zeros(images[image, :, :].shape)

            for lbl in valid_lbls:
                result = np.logical_or(result, (image_labeled == lbl))

                images[image, :, :] = result

    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

    return images

def results_evaluation(results, test_GT, prints):
    # Evaluation sklearn
    if prints:
        sys.stdout.write('Evaluating results... ')
        t = time.time()
    metrics = ev.getMetrics(test_GT, results)

    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n\n')

        print "Recall: " + str(metrics[0] * 100)
        print "Precision: " + str(metrics[1] * 100)
        print "F1: " + str(metrics[2] * 100)

    return metrics

def f1_p(train, test, test_GT, alpha, ro, conn):
    prints = False
    # F1-score vs #Pixels
    p_range = np.around(np.arange(0,100,10))

    metrics_array = []

    results = background_substraction(train, test, alpha, ro, prints)

    results = hole_filling(results, conn, prints)

    for p in p_range:
        sys.stdout.write("(p=" + str(p)+")\n")

        results_filtered = area_filtering(results, conn, p, prints)

        metrics = results_evaluation(results_filtered, test_GT, prints)

        metrics_array.append(metrics);

    # TASK 2.2 - Plot F1-score vs Alpha
    x = p_range
    metrics_array = np.array(metrics_array)
    y = metrics_array[:, 2]

    f1_max = np.max(metrics_array[:, 2])
    f1_max_idx = np.argmax(metrics_array[:, 2])
    best_p = p_range[f1_max_idx]

    print "F1: " + str(np.around(f1_max,decimals=4)) + " (p="+str(best_p)+")"

    axis = ["#Pixels", "F1-score"]
    labels = []
    ev.plotGraphics(x, y, axis, labels)

def auc(results, test_GT):
    sys.stdout.write('Computing Precision-Recall curve... ')
    # Background substraction
    g = GaussianModelling(adaptive_ratio=0.15, grayscale_modelling=False)
    g.fit(train)
    scores = g.predict_probabilities(test)

    t = time.time()
    precision, recall, auc_val = ev.getPR_AUC(test_GT, scores)
    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')

    plt.step(recall, precision, color='g', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='g')
    print "AUC: " + str(auc_val)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    # plt.title("Precision-Recall curve (AUC=" + str(auc_val) + ")" )
    plt.title("Precision-Recall curve - Fall")
    plt.show()



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

if __name__ == "__main__":
    main()
