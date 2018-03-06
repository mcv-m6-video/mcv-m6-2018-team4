import numpy as np
import cv2
import matplotlib.pyplot as plt


class GaussianModelling:
    def __init__(self, alpha=.05, adaptive_ratio=0, grayscale_modelling=True):
        self.alpha = alpha
        self.adaptive_ratio = adaptive_ratio
        self.grayscale_modelling = grayscale_modelling

    def fit(self, X):
        # If input images are in BGR color and grayscale modelling is enabled
        if np.shape(X)[-1] == 3 and self.grayscale_modelling:
            # Save gray images in a new array
            Xorig = X
            X = np.empty(np.shape(X)[:3])

            # Convert from BGR to GRAY
            for i in range(len(Xorig)):
                X[i] = cv2.cvtColor(Xorig[i], cv2.COLOR_BGR2GRAY)

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
            # If input images are in BGR color and grayscale modelling is enabled
            if np.shape(X)[-1] == 3 and self.grayscale_modelling:
                # Convert to grayscale
                im = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)
                # Segment frame
                y[i] = np.abs(im - self.mean) >= self.alpha * (self.std + 2)
            else:
                im = X[i]
                # Segment frame
                y[i] = np.all(np.abs(im - self.mean) >= self.alpha * (self.std + 2), axis=2)

            # Adapt the mean and std (Adaptive Gaussian Modelling)
            self.mean = (1 - self.adaptive_ratio) * self.mean + self.adaptive_ratio * im
            self.std = np.sqrt(
                (1 - self.adaptive_ratio) * np.power(self.std, 2) + self.adaptive_ratio * np.power(im - self.mean, 2))

            # plt.imshow(y[i],cmap='gray')
            # plt.show()

        return y

    def predict_probabilities(self, X):
        y = np.empty(np.shape(X)[:3])
        for i in range(len(X)):
            # If input images are in BGR color and grayscale modelling is enabled
            if np.shape(X)[-1] == 3 and self.grayscale_modelling:
                # Convert to grayscale
                im = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)
                # Segment frame
                y[i] = np.abs(im - self.mean) / (self.std + 2)
            else:
                im = X[i]
                # Segment frame
                y[i] = np.all(np.abs(im - self.mean) / (self.std + 2), axis=2)

                # Adapt the mean and std (Adaptive Gaussian Modelling)
            self.mean = (1 - self.adaptive_ratio) * self.mean + self.adaptive_ratio * im
            self.std = np.sqrt(
                (1 - self.adaptive_ratio) * np.power(self.std, 2) + self.adaptive_ratio * np.power(im - self.mean, 2))

            # plt.imshow(y[i],cmap='gray')
            # plt.show()

        return y
