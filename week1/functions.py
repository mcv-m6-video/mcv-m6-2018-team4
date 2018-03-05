import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def ConfusionMatrix(GT, Prediction):
    if len(GT) != len(Prediction):
        print "Datasets does not have same size"
        return -1
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    TPFv = []
    TPGTv = []
    F1v = []

    print "Computing Confusion Matrix"
    for img in range(len(GT)):
        GT_img = np.array(GT[img])
        Prediction_img = np.array(Prediction[img])
        print "Computing image: " + str(img)
        TPF = 0
        TNF = 0
        FPF = 0
        FNF = 0
        TPFGT = 0
        for i in range(GT_img.shape[0]):
            for j in range(GT_img.shape[1]):
                GT_value = GT_img[i][j][0]
                Prediction_value = Prediction_img[i][j][0]
                if GT_value >= 170:
                    GT_value = 1
                    TPFGT += 1
                else:
                    GT_value = 0
                if 1 == Prediction_value and GT_value == 1:
                    TP += 1
                    TPF += 1
                elif 0 == Prediction_value and GT_value == 0:
                    TN += 1
                    TNF += 1
                elif 1 == Prediction_value and GT_value == 0:
                    FP += 1
                    FPF += 1
                elif 0 == Prediction_value and GT_value == 1:
                    FN += 1
                    FNF += 1

        AF, RF, PF, F1F = Metrics(TPF, TNF, FPF, FNF, False)
        TPFv.append(TPF)
        TPGTv.append(TPFGT)
        F1v.append(F1F)
    return [TP, TN, FP, FN], TPFv, TPGTv, F1v

def ConfusionMatrixDesync(GT, Prediction):
    if len(GT) != len(Prediction):
        print "Datasets does not have same size"
        return -1
    vDelay = []
    for delay in range(0,21,5):
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        TPFv = []
        TPGTv = []
        F1v = []

        print "Computing Confusion Matrix Delay " + str(delay)
        for img in range(len(GT)-delay):
        # for img in range(10):
            GT_img = np.array(GT[img])
            Prediction_img = np.array(Prediction[img+delay])
            print "Computing image: " + str(img)
            TPF = 0
            TNF = 0
            FPF = 0
            FNF = 0
            TPFGT = 0
            for i in range(GT_img.shape[0]):
                for j in range(GT_img.shape[1]):
                    GT_value = GT_img[i][j][0]
                    Prediction_value = Prediction_img[i][j][0]
                    if GT_value >= 170:
                        GT_value = 1
                        TPFGT += 1
                    else:
                        GT_value = 0
                    if 1 == Prediction_value and GT_value == 1:
                        TP += 1
                        TPF += 1
                    elif 0 == Prediction_value and GT_value == 0:
                        TN += 1
                        TNF += 1
                    elif 1 == Prediction_value and GT_value == 0:
                        FP += 1
                        FPF += 1
                    elif 0 == Prediction_value and GT_value == 1:
                        FN += 1
                        FNF += 1

            AF, RF, PF, F1F = Metrics(TPF, TNF, FPF, FNF, False)
            TPFv.append(TPF)
            TPGTv.append(TPFGT)
            F1v.append(F1F)
        vDelay.append([[TP, TN, FP, FN], TPFv, TPGTv, F1v])

    return vDelay

def Metrics(TP, TN, FP, FN, show=True):
    if show:
        print "Computing Metrics with TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
    Accuracy = (float(TP) + float(TN)) / (float(TP) + float(TN) + float(FP) + float(FN))
    Recall = float(TP) / (float(TP) + float(FN))

    if ((float(TP) + float(FP)) > 0):
        Precision = float(TP) / (float(TP) + float(FP))
    else:
        Precision = 0

    if ((float(Precision) + float(Recall)) > 0):
        F1 = 2 * float(Precision) * float(Recall) / (float(Precision) + float(Recall))
    else:
        F1 = 0

    if show:
        print "Accuracy: " + str(Accuracy * 100)
        print "Recall: " + str(Recall * 100)
        print "Precision: " + str(Precision * 100)
        print "F1: " + str(F1 * 100)
    return Accuracy, Recall, Precision, F1

def plotGraphics(x, y, axis, labels):
    for i in range(len(labels)):
        plt.plot(x[i], y[i], label=labels[i])
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    plt.legend()
    plt.show()

def showMultipleGraphic(x, y, label, labelx1, labelx2):
    plt.plot(x[0], y[0], label=labelx1)
    plt.plot(x[1], y[1], label=labelx2)
    plt.xlabel("time")
    plt.ylabel(label)
    plt.show()

def showSimpleGraphic(x, y, label):
    plt.plot(x, y)
    plt.xlabel("time")
    plt.ylabel(label)
    plt.show()
