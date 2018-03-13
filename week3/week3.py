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
from sklearn.metrics import auc

import sys
import matplotlib.pyplot as plt
from matplotlib import animation

from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import binary_fill_holes, generate_binary_structure, label, labeled_comprehension
from skimage.morphology import remove_small_objects

def main():
    # Week 2 Best configurations
    #   - Highway   (a=2.4,ro=0.15) F1=0.6319 - conn4 (65.3461132183114)
    #   - Fall      (a=3.2,ro=0.05) F1=0.6896 - conn4 (73.94992293059627)
    #   - Traffic   (a=3.5,ro=0.15) F1=0.6376 - conn8 (63.954441547032424) conn4(63.91668494564192)

    # Week 2 Best configurations
    #   - Highway   (a=2.4,ro=0.15) AUC=0.358 - conn4 (0.5102)  conn8(0.5081)
    #   - Fall      (a=3.2,ro=0.05) AUC=0.689 - conn4 (0.6968)  conn8(0.6893)
    #   - Traffic   (a=3.5,ro=0.15) AUC=0.497 - conn4 (0.5515)  conn8(0.5509)


    dataset_name = 'highway'
    # dataset_name = 'fall'
    # dataset_name = 'traffic'

    if dataset_name == 'highway':
        frames_range = (1051, 1350)
        # alpha = 2.4
        alpha = 1.8
        ro = 0.15
        p = 220
        conn = 4


    elif dataset_name == 'fall':
        frames_range = (1461, 1560)
        # alpha = 3.2
        alpha = 1.4
        ro = 0.05
        p = 1800
        conn = 8

    elif dataset_name == 'traffic':
        frames_range = (951, 1050)
        # alpha = 3.5
        alpha = 2.8
        ro = 0.15
        p = 330
        conn = 4

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
    # results, metrics = task1_pipeline(train, test, test_GT, alpha, ro, 8, True)
    # precision_recall_curve(train, test, test_GT, ro, conn, p)

    # TASK 2 - Area Filtering
    # results, metrics = task2_pipeline(train, test, test_GT, alpha, ro, conn, p)

    # f1_p(train, test, test_GT, alpha, ro, 4)
    # auc_vs_p(train, test, test_GT, ro, 4)
    precision_recall_curve(train, test, test_GT, ro, conn, p)
    # f1score_alpha(train, test, test_GT, conn, ro, p, prints=False)

    # auc_all()

    # TASK 3 - Morphology

    # TASK 4 - Shadow removal

    # make_gif(results)

    # TASK 5 - Show improvements (PR-curve/AUC)

def task1_pipeline(train, test, test_GT, alpha, ro, conn, prints=True):

    results = background_substraction(train, test, alpha, ro)
    results = hole_filling(results,conn)
    metrics = results_evaluation(results, test_GT)
    return results, metrics

def task2_pipeline(train, test, test_GT, alpha, ro, conn, p, prints=True):

    results = background_substraction(train, test, alpha, ro, prints)
    results = hole_filling(results,conn, prints)
    results = area_filtering(results, p, prints)
    metrics = results_evaluation(results, test_GT, prints)

    return results, metrics

def background_substraction(train, test, alpha, ro, prints=True):

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

def hole_filling(images, conn, prints=True):

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

def make_gif(results):
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

def area_filtering(images, pixels, prints):

    if prints:
        t = time.time()
        sys.stdout.write('Computing area filtering... ')

    results = np.zeros(images.shape)

    for image in range(len(images)):
        results[image, :, :] = remove_small_objects(images[image, :, :].astype(np.bool), pixels)

    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

    return results

def results_evaluation(results, test_GT, prints=True):
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

def auc_vs_p(train, test, test_GT, ro, conn):
    # auc vs #Pixels
    p_range = np.around(np.arange(0, 1000, 10))

    auc_array = []

    for p in p_range:
        sys.stdout.write("(p=" + str(p)+") ")
        auc = precision_recall_curve(train, test, test_GT, ro, conn, p, False)
        auc_array.append(auc);

    # TASK 2.2 - Plot F1-score vs Alpha
    x = p_range
    y = np.array(auc_array)

    auc_max = np.max(y)
    auc_max_idx = np.argmax(y)
    best_p = p_range[auc_max_idx]

    print "AUC: " + str(np.around(auc_max,decimals=4)) + " (p="+str(best_p)+")"

    axis = ["#Pixels", "AUC"]
    labels = []
    ev.plotGraphics(x, y, axis, labels)

    print y

def precision_recall_curve(train, test, test_GT, ro, conn, p, prints=True):
    tt = time.time()
    sys.stdout.write('Computing Precision-Recall curve... ')

    alpha_range = np.around(np.arange(0, 16.2, 1), decimals=2)

    metrics_array = []

    for alpha in alpha_range:
        if prints:
            t = time.time()
            sys.stdout.write("(alpha=" + str(np.around(alpha, decimals=2)) + ") ")
        results = background_substraction(train, test, alpha, ro, False)
        results = hole_filling(results, conn, False)
        results = area_filtering(results, p, False)
        metrics = results_evaluation(results, test_GT, False)

        metrics_array.append(metrics)

        if prints:
            elapsed = time.time() - t
            sys.stdout.write(str(elapsed) + ' sec \n')


    precision = np.array(metrics_array)[:, 0]
    recall = np.array(metrics_array)[:, 1]
    auc_val = auc(recall, precision)

    sys.stdout.write("(auc=" + str(np.around(auc_val, decimals=4)) + ") ")

    elapsed = time.time() - tt
    sys.stdout.write(str(elapsed) + ' sec \n')


    if prints:
        plt.plot(recall, precision, color='g')
        print "AUC: "+ str(auc_val)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title("Precision-Recall curve (AUC=" + str(auc_val) + ")" )
        # plt.title("Precision-Recall curve - Fall" )
        plt.show()

    return auc_val

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

def f1score_alpha(train, test, test_GT, conn, ro, p, prints=False):
    # Task 1.2 - F1-score vs Alpha
    alpha_range = np.around(np.arange(1,5,0.2),decimals=2)

    metrics_array = []

    for alpha in alpha_range:
        sys.stdout.write("(alpha=" + str(np.around(alpha,decimals=2))+") ")
        t= time.time()

        results = background_substraction(train, test, alpha, ro, prints)
        results = hole_filling(results, conn, prints)
        results = area_filtering(results, p, prints)
        metrics = results_evaluation(results, test_GT, prints)

        metrics_array.append(metrics);

        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

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

def auc_all():
    # p_range = np.concatenate([np.array([0]),np.around(np.arange(0, 1000, 10))])
    p_range = np.around(np.arange(0, 1000, 10))
    # p_range2 = np.concatenate([np.array([0]),np.around(np.arange(0, 2000, 50))])
    p_range2 = np.around(np.arange(0, 2000, 50))
    x = [p_range, p_range,p_range2]

    y_traffic = [0.54865382, 0.5592443,  0.56580872, 0.571033,   0.57604746, 0.58004003,
 0.5837208,  0.58664876, 0.58901124, 0.59100597, 0.59249389, 0.59403019,
 0.59510371, 0.59685786, 0.59798599, 0.5998437,  0.60081657, 0.60195222,
 0.60377896, 0.60619143, 0.6073376,  0.60800021, 0.60910401, 0.60971082,
 0.61069558, 0.61044473, 0.61120665, 0.61254456, 0.61291559, 0.61321382,
 0.6133638,  0.61385636, 0.61263136, 0.61559219, 0.61488785, 0.61297316,
 0.61356487, 0.61378428, 0.61202322, 0.61133323, 0.60994398, 0.61029301,
 0.61085772, 0.61006678, 0.60799569, 0.60644023, 0.60678493, 0.60822458,
 0.6077443,  0.60867033, 0.60805301, 0.60819197, 0.6092698,  0.60974066,
 0.60882509, 0.60669797, 0.60818745, 0.60718756, 0.60773519, 0.60776267,
 0.6078122, 0.60824679, 0.6095641,  0.60933804, 0.60776864, 0.60615064,
 0.60669904, 0.60632972, 0.60869301, 0.60922683, 0.60929263, 0.61018956,
 0.60990093, 0.61049604, 0.61015858, 0.61081188, 0.61023095, 0.60949348,
 0.60957899, 0.60861016, 0.60771793, 0.60772563, 0.60574544, 0.60728,
 0.60641079, 0.60923875, 0.60942574, 0.60876463, 0.60908811, 0.60917016,
 0.60714994, 0.60661957, 0.60623696, 0.60641611, 0.60608787, 0.60609832,
 0.60649841, 0.60685467, 0.60777793, 0.60863877]

    y_highway = [0.47646952, 0.51264552, 0.51712399, 0.52016654, 0.52226093, 0.52292387,
 0.52324357, 0.52385192, 0.524463,   0.52590177, 0.52727928, 0.52689404,
 0.52689589, 0.52699987, 0.52692316, 0.52722481, 0.52747004, 0.52697665,
 0.52713471, 0.52716288, 0.52698196, 0.52691934, 0.52779482, 0.52722408,
 0.52739301, 0.52662726, 0.52663605, 0.52684491, 0.52615045, 0.52613233,
 0.52673217, 0.52655387, 0.52701599, 0.52699287, 0.52696298, 0.52633702,
 0.52651062, 0.52572597, 0.5247644,  0.52461462, 0.52392415, 0.52282114,
 0.52152874, 0.52096475, 0.52080267, 0.52094456, 0.52068838, 0.51971823,
 0.51961159, 0.51939336, 0.52018412, 0.52096747, 0.52012205, 0.51897652,
 0.5194545,  0.51901482, 0.51966423, 0.5194988,  0.51878067, 0.51862623,
 0.51815943, 0.51723116, 0.51618216, 0.51642436, 0.51654179, 0.51581763,
 0.51608789, 0.51520603, 0.5162197,  0.51618521, 0.51574108, 0.51559478,
 0.51431996, 0.51502,    0.51515404, 0.5150161,  0.51474541, 0.51451247,
 0.51444362, 0.51442594, 0.51382708, 0.51312868, 0.51342386, 0.51360329,
 0.51245204, 0.5115633,  0.51205942, 0.5110433,  0.51141787, 0.51271048,
 0.5133746,  0.51237951, 0.51229149, 0.51209891, 0.51304629, 0.51176097,
 0.51267674, 0.51321841, 0.51307787, 0.5136463 ]

    y_fall = [0.60677031, 0.70924287, 0.73032806, 0.7406121,  0.7490444,  0.75348709,
 0.76354178, 0.77049525, 0.7728623,  0.77931308, 0.78337424, 0.78399852,
 0.78696076, 0.79017504, 0.79253606, 0.79500976, 0.79990754, 0.80621057,
 0.81470682, 0.81729479, 0.81780651, 0.81886046, 0.82048388, 0.82309066,
 0.82532147, 0.82611948, 0.8287993,  0.82972555, 0.83358859, 0.83771235,
 0.83976428, 0.84408887, 0.84612412, 0.84775577, 0.84893319, 0.84928296,
 0.85109461, 0.851044, 0.85095705, 0.85093486]

    # y = [y_traffic, y_highway, y_fall]
    x = p_range
    y = y_traffic
    axis = ["#Pixels", "AUC"]
    # labels = ["Traffic", "Highway", "Fall"]
    labels = []
    ev.plotGraphics(x, y, axis, labels)

# Other old functions
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

def area_filtering2(images, conn, pixels, prints):

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

if __name__ == "__main__":
    main()
