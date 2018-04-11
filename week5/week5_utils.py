import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import binary_fill_holes, generate_binary_structure
from skimage.morphology import remove_small_objects

sys.path.append('..')
sys.path.append('../week2/')
sys.path.append('../week3/')
sys.path.append('../week4/')

from gaussian_modelling import GaussianModelling
import morphology as morph
import evaluation as ev

def mask_pipeline(train, test, test_GT, alpha, ro, conn, p, dataset, prints=True, ROI=None, valid_pixels=None):

    results = background_substraction(train, test, alpha, ro, prints)
    results = hole_filling(results,conn, prints)

    if ROI != None:
        for i in range(len(results)):
            results[i][ROI[0]] = 0

    if valid_pixels != None:
        for i in range(len(results)):
            results[i][valid_pixels[i]] = 0

    results = area_filtering(results, p, prints)

    # Morphology
    if dataset == 'highway':
        results = morphology_highway(results, conn, prints=True)

    elif dataset == 'traffic':
        results = morphology_traffic(results, conn, prints=True)

    elif dataset == 'traffic_stab':
        results = morphology_traffic_stab(results, conn, prints=True)

    elif dataset == 'sequence_parc_nova_icaria':
        results = morphology_traffic(results, conn, prints=True)

    elif dataset == 'week5_dataset':
        results = morphology_week5(results, conn, prints=True)
    else:
        print "Invalid dataset name"
        return

    if test_GT == []:
        metrics = [0]
    else:
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

def morphology_traffic(images, conn, prints=True):

    if prints:
        t = time.time()
        sys.stdout.write('Computing morphology... ')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    results = morph.Closing(images, kernel, False)

    results = hole_filling(results,conn, False)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    results = morph.Opening(results, kernel, False)

    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

    return results

def morphology_traffic_stab(images, conn, prints=True):

    if prints:
        t = time.time()
        sys.stdout.write('Computing morphology... ')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    results = morph.Closing(images, kernel, False)

    results = hole_filling(results,conn, False)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 22))
    results = morph.Opening(results, kernel, False)

    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

    return results

def morphology_highway(images, conn, prints=True):

    if prints:
        t = time.time()
        sys.stdout.write('Computing morphology... ')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    results = morph.Closing(images, kernel, False)

    results = hole_filling(results,conn, False)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    results = morph.Opening(results, kernel, False)

    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

    return results


def morphology_week5(images, conn, prints=True):

    if prints:
        t = time.time()
        sys.stdout.write('Computing morphology... ')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    results = morph.Closing(images, kernel, False)

    results = hole_filling(results,conn, False)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    results = morph.Opening(results, kernel, False)

    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

    return results

def results_evaluation(results, test_GT, prints=True):
    # Evaluation sklearn
    if prints:
        sys.stdout.write('\nEvaluating results... ')
        t = time.time()
    metrics = ev.getMetrics(test_GT, results)

    if prints:
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

        print "\t - Recall: " + str(metrics[0] * 100)
        print "\t - Precision: " + str(metrics[1] * 100)
        print "\t - F1: " + str(metrics[2] * 100)
        print ""

    return metrics

def make_gif_mask(results):
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

def make_gif(results, gifname):

    sys.stdout.write('Making GIF... ')
    t = time.time()

    N = len(results)
    # H, W, C = results[0].shape

    ims = []
    fig = plt.figure()
    for i in range(N):
        im = plt.imshow(cv2.cvtColor(results[i].astype(np.uint8), cv2.COLOR_BGR2RGB),animated=True)
        plt.axis("off")
        ims.append([im])

    anim = animation.ArtistAnimation(fig, ims, interval=len(results), blit=True)
    anim.save(gifname, writer='imagemagick', fps=5)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n\n')

    plt.show()

    return