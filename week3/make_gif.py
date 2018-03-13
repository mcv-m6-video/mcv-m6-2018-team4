import sys

sys.path.append('..')
sys.path.append('../week2/')

from week3 import hole_filling, background_substraction

import cv2
import evaluation as ev
import numpy as np
import os
import time
from dataset import Dataset
from gaussian_modelling import GaussianModelling
from matplotlib import animation
import matplotlib.pyplot as plt


def main():
    # dataset_name = 'highway'
    dataset_name = 'fall'
    # dataset_name = 'traffic'

    if dataset_name == 'highway':
        frames_range = (1051, 1350)
        alpha = 2.4
        ro = 0.15

    elif dataset_name == 'fall':
        frames_range = (1461, 1560)
        alpha = 3.2
        ro = 0.05

    elif dataset_name == 'traffic':
        frames_range = (951, 1050)
        alpha = 3.5
        ro = 0.15

    else:
        print "Invalid dataset name"

    # Read dataset
    dataset = Dataset(dataset_name, frames_range[0], frames_range[1])

    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    # Split dataset
    train = imgs[:len(imgs) / 2]
    test = imgs[len(imgs) / 2:]
    test_GT = imgs_GT[len(imgs) / 2:]

    results = background_substraction(train, test, alpha, ro, prints=True)
    copy_of_results = np.copy(results)
    results_hf = hole_filling(copy_of_results, 4, prints=True)

    im_plot_list = []
    fig = plt.figure()
    for i in range(len(results)):
        im = paint_in_image(results[i], color='white')
        im = paint_in_image(results_hf[i] - results[i], im, color=(255, 255, 0))

        im_plot = plt.imshow(im, animated=True)
        plt.axis("off")
        im_plot_list.append([im_plot])

    anim = animation.ArtistAnimation(fig, im_plot_list, interval=len(results), blit=True)
    anim.save('animation.gif', writer='imagemagick', fps=10)
    plt.show()


def paint_in_image(mask, image=None, color='white'):
    if color == 'white':
        channel_values = (255, 255, 255)
    elif color == 'red':
        channel_values = (255, 0, 0)
    elif color == 'green':
        channel_values = (0, 255, 0)
    elif color == 'blue':
        channel_values = (0, 0, 255)
    elif len(color) == 3 and isinstance(color[0], int):
        channel_values = color
    else:
        print('{} not a valid color'.format(color))

    if image is None:
        image = np.zeros([mask.shape[0], mask.shape[1], 3], dtype=np.uint8)
    else:
        image = np.copy(image)

    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    red_channel[mask == 1] = channel_values[0]
    green_channel[mask == 1] = channel_values[1]
    blue_channel[mask == 1] = channel_values[2]

    return image


if __name__ == "__main__":
    main()
