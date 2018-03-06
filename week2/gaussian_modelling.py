import numpy as np
import cv2
import matplotlib.pyplot as plt


class GaussianModelling:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X):
        # Convert from BGR to GRAY
        if np.shape(X)[-1] == 3:
            for i in range(len(X)):
                X[i] = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)

        # Compute mean and std
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X - self.mean, axis=0)

        # plt.imshow(self.mean,cmap='gray')
        # plt.figure()
        # plt.imshow(self.std,cmap='gray')
        # plt.show()

        return self

    def predict(self, X):
        y = np.empty(np.shape(X))
        for i in range(len(X)):
            # Convert frame from BGR to GRAY
            if X[i].shape[-1] == 3:
                X[i] = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)

            # Segment frame
            y[i] = np.abs(X[i] - self.mean) >= self.alpha * (self.std + 2)

            # plt.imshow(y[i],cmap='gray')
            # plt.show()

        return y
