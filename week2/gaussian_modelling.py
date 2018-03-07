import warnings

import numpy as np
import cv2
import matplotlib.pyplot as plt


class GaussianModelling:
    def __init__(self, alpha=1.8, adaptive_ratio=0, grayscale_modelling=True, hsv=False):
        self.alpha = alpha
        self.adaptive_ratio = adaptive_ratio
        self.grayscale_modelling = grayscale_modelling
        self.hsv = hsv
        if self.grayscale_modelling and self.hsv:
            print 'WARNING: HSV parameter ignored if Grayscale Modelling is enabled.'

    def fit(self, X):
        # If input images are in BGR color and grayscale or HSV modelling is enabled
        if np.shape(X)[-1] == 3 and (self.grayscale_modelling or self.hsv):
            # Save images in a new array
            Xorig = X
            if self.grayscale_modelling:
                X = np.empty(np.shape(X)[:3])
                cspace = cv2.COLOR_BGR2GRAY
            else:
                X = np.empty(np.shape(X))
                cspace = cv2.COLOR_BGR2HSV

            # Convert colorspace
            for i in range(len(Xorig)):
                X[i] = cv2.cvtColor(Xorig[i], cspace)

        # Compute mean and std
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X - self.mean, axis=0)

        # plt.imshow(self.mean,cmap='gray')
        # plt.figure()
        # plt.imshow(self.std,cmap='gray')
        # plt.show()

        return self

    def predict(self, X):
        y = np.empty(np.shape(X)[:3])
        for i in range(len(X)):
            # If input images are in BGR color and grayscale or HSV modelling is enabled
            if np.shape(X)[-1] == 3 and (self.grayscale_modelling or self.hsv):
                # Convert image
                cspace = cv2.COLOR_BGR2GRAY if self.grayscale_modelling else cv2.COLOR_BGR2HSV
                im = cv2.cvtColor(X[i], cspace)
            else:
                im = X[i]

            # Segment image
            aux = np.abs(im - self.mean) >= self.alpha * (self.std + 2)

            # If it is a color image apply AND over the channels
            if np.shape(aux)[-1] == 3:
                aux = np.all(aux, axis=2)

            y[i] = aux

            mean_flat = self.mean.flatten()
            std_flat = self.std.flatten()
            bck_px= y[i].flatten()<0.5
            # Adapt the mean and std (Adaptive Gaussian Modelling)
            mean_flat[bck_px] = \
                (1 - self.adaptive_ratio) * mean_flat[bck_px] + self.adaptive_ratio * im.flatten()[bck_px]
            std_flat[bck_px] = np.sqrt(
                (1 - self.adaptive_ratio) * np.power(std_flat[bck_px], 2) + self.adaptive_ratio * np.power(
                    (im.flatten()[bck_px] - mean_flat[bck_px]), 2))

            self.mean = np.resize(mean_flat,np.shape(self.mean))
            self.std = np.resize(std_flat,np.shape(self.std))

            # plt.imshow(y[i],cmap='gray')
            # plt.show()

        return y

    def predict_probabilities(self, X):
        if not self.grayscale_modelling:
            print 'ERROR: Probabilities only computable on grayscale modelling.'
            exit(-1)

        y = np.empty(np.shape(X)[:3])
        for i in range(len(X)):
            # If input images are in BGR color and grayscale modelling is enabled
            if np.shape(X)[-1] == 3 and self.grayscale_modelling:
                # Convert to grayscale
                im = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)

            # Segment frame
            y[i] = np.abs(im - self.mean) / (self.std + 2)

            # Adapt the mean and std (Adaptive Gaussian Modelling)
            mean_flat = self.mean.flatten()
            std_flat = self.std.flatten()
            bck_px = y[i].flatten() < 0.5

            mean_flat[bck_px] = \
                (1 - self.adaptive_ratio) * mean_flat[bck_px] + self.adaptive_ratio * im.flatten()[bck_px]
            std_flat[bck_px] = np.sqrt(
                (1 - self.adaptive_ratio) * np.power(std_flat[bck_px], 2) + self.adaptive_ratio * np.power(
                    (im.flatten()[bck_px] - mean_flat[bck_px]), 2))

            self.mean = np.resize(mean_flat, np.shape(self.mean))
            self.std = np.resize(std_flat, np.shape(self.std))

            # plt.imshow(y[i],cmap='gray')
            # plt.show()

        return y
