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

def morphology_traffic(train, test, test_GT, alpha, ro, conn, p, prints=True, valid_pixels=None):

    results = background_substraction(train, test, alpha, ro, prints)
    results = hole_filling(results,conn, prints)

    if valid_pixels != None:
        for i in range(len(results)):
            results[i][valid_pixels[i]] = 0

    results = area_filtering(results, p, prints)

    # Morphology start
    t = time.time()
    sys.stdout.write('Computing morphology... ')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    results = morph.Closing(results, kernel, False)

    results = hole_filling(results,conn, False)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    results = morph.Opening(results, kernel, False)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')

    metrics = results_evaluation(results, test_GT, prints)

    return results, metrics

def morphology_highway(train, test, test_GT, alpha, ro, conn, p, prints=True, valid_pixels=None):

    results = background_substraction(train, test, alpha, ro, prints)
    results = hole_filling(results,conn, prints)

    if valid_pixels != None:
        for i in range(len(results)):
            results[i][valid_pixels] = 0

    results = area_filtering(results, p, prints)

    # Morphology start
    t = time.time()
    sys.stdout.write('Computing morphology... ')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    results = morph.Closing(results, kernel, False)

    results = hole_filling(results,conn, False)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    results = morph.Opening(results, kernel, False)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')

    metrics = results_evaluation(results, test_GT, prints)

    return results, metrics

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
    anim.save(gifname, writer='imagemagick', fps=20)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n\n')

    plt.show()

    return