import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, auc


def readImages(Path, extension):
    if extension == "" or (extension != "jpg" and extension != "png") or not os.path.exists(Path):
        print "Extension or path is invalid"
        return -1
    if len(os.listdir(Path)) == 0:
        print "Directory is empty"
        return 0

    imgs = []
    for file in sorted(os.listdir(Path)):
        if file.endswith("." + extension):
            print "loading image " + os.path.join(Path, file)
            image = cv2.imread(os.path.join(Path, file))
            imgs.append(image)
    return imgs


def confusionMatrix(GT, Prediction):
    test_GT_v = np.array(GT)[:, :, :, 0].flatten()
    valid_pixels = np.nonzero(np.any([(test_GT_v != 85), (test_GT_v != 170)], axis=0))
    y_true = (test_GT_v == 255)

    y_predicted = np.array(Prediction).flatten()

    matrix = confusion_matrix(y_true[valid_pixels], y_predicted[valid_pixels])

    return matrix


def getMetrics(GT, Prediction):
    test_GT_v = np.array(GT)[:, :, :, 0].flatten()
    valid_pixels = np.nonzero(np.any([(test_GT_v != 85), (test_GT_v != 170)], axis=0))
    y_true = (test_GT_v == 255)

    y_predicted = np.array(Prediction).flatten()
    # y_predicted = (y_predicted == 255) # Descoment for SOTA

    metrics = precision_recall_fscore_support(y_true[valid_pixels], y_predicted[valid_pixels], average="binary",
                                              pos_label=1)

    return metrics


def getPR_AUC(GT, Score):
    test_GT_v = np.array(GT)[:, :, :, 0].flatten()
    valid_pixels = np.nonzero(np.any([(test_GT_v != 85), (test_GT_v != 170)], axis=0))
    y_true = (test_GT_v == 255)

    y_score = np.array(Score).flatten()

    precision, recall, _ = precision_recall_curve(y_true[valid_pixels], y_score[valid_pixels])
    auc_val = auc(recall, precision)

    return precision, recall, auc_val


def plotGraphics(x, y, axis, labels):
    if len(labels) > 1:
        for i in range(len(labels)):
            plt.plot(x[i], y[i], label=labels[i])
    else:
        plt.plot(x, y, label=labels)

    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    if len(labels) > 0:
        plt.legend()
    plt.show()
