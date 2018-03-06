import sys
sys.path.append('..')
import cv2
import evaluation as ev
import numpy as np
import os
import time
from dataset import Dataset
from gaussian_modelling import GaussianModelling
from matplotlib import animation
import matplotlib.pyplot as plt
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
    ims = []
    fig=plt.figure()
    for i in range(len(results)):
        im = plt.imshow(results[i],cmap='gray',animated=True)
        plt.axis("off")
        ims.append([im])

    anim = animation.ArtistAnimation(fig, ims,interval=len(results), blit=True)
    anim.save('animation.gif', writer='imagemagick', fps=10)
    plt.show()



if __name__ == "__main__":
    main()
