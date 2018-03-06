import numpy as np
import cv2
import matplotlib.pyplot as plt


class GaussianModelling:
    def __init__(self, alpha=.05, adaptive_ratio=0):
        self.alpha = alpha
        self.adaptive_ratio = adaptive_ratio

    def fit(self, X):
        # If input images are in BGR color
        if np.shape(X)[-1] == 3:
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
            # Convert frame from BGR to GRAY
            if X[i].shape[-1] == 3:
                im = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)
            else:
                im = X[i]

            # Segment frame
            y[i] = np.abs(im - self.mean) >= self.alpha * (self.std + 2)

            # Adapt the mean and std (Adaptive Gaussian Modelling)
            self.mean = (1 - self.adaptive_ratio) * self.mean + self.adaptive_ratio * im
            self.std = (1 - self.adaptive_ratio) * self.std + self.adaptive_ratio * np.abs(im - self.mean)
            # plt.clf()
            #plt.imshow(y[i],cmap='gray')
            #plt.hold(True)
            #plt.show()

        return y
