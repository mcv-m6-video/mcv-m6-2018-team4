import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_auc_score, \
    auc
import sys
from gaussian_modelling import GaussianModelling
import time

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
    valid_pixels = np.nonzero(np.any([(test_GT_v!= 85),(test_GT_v!= 170)], axis=0))
    y_true = (test_GT_v == 255)

    y_predicted = np.array(Prediction).flatten()

    matrix = confusion_matrix(y_true[valid_pixels], y_predicted[valid_pixels])

    return matrix

def getMetrics(GT, Prediction):
    test_GT_v = np.array(GT)[:, :, :, 0].flatten()
    valid_pixels = np.nonzero(np.any([(test_GT_v!= 85),(test_GT_v!= 170)], axis=0))
    y_true = (test_GT_v == 255)

    y_predicted = np.array(Prediction).flatten()
    # y_predicted = (y_predicted == 255) # Descoment for SOTA

    metrics = precision_recall_fscore_support(y_true[valid_pixels], y_predicted[valid_pixels], average="binary",pos_label=1)

    return metrics

def getPR_AUC(GT, Score):
    test_GT_v = np.array(GT)[:, :, :, 0].flatten()
    valid_pixels = np.nonzero(np.any([(test_GT_v!= 85),(test_GT_v!= 170)], axis=0))
    y_true = (test_GT_v == 255)

    y_score = np.array(Score).flatten()

    precision, recall, _ = precision_recall_curve(y_true[valid_pixels], y_score[valid_pixels])
    auc_val = auc(recall, precision)

    return precision, recall, auc_val

def plotGraphics(x, y, axis, labels):
    if len(labels)>1:
        for i in range(len(labels)):
            plt.plot(x[i], y[i], label=labels[i])
    else:
        plt.plot(x, y, label=labels)

    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    if len(labels)>0:
        plt.legend()
    plt.show()





# def ConfusionMatrix(GT, Prediction, show=False):
#     if len(GT) != len(Prediction):
#         print "Datasets does not have same size"
#         return -1
#
#     img_shape = np.shape(Prediction[0])
#     conf_matrix = np.zeros((2,2,len(Prediction)))
#
#     sys.stdout.write("computing confusion matrix...")
#     # sys.stdout.flush()
#     for img in range(len(GT)):
#         GT_img = np.array(GT[img])
#         Prediction_img = np.array(Prediction[img])
#
#         if show:
#             print "Computing image: " + str(img)
#
#         # Prediction_img = Prediction_img[:,:,0]
#         Prediction_img_v = Prediction_img.reshape((img_shape[0]*img_shape[1]))
#         GT_img = GT_img[:,:,0]
#         GT_img_v = GT_img.reshape((img_shape[0]*img_shape[1]))
#
#         # valid_pixels = np.nonzero(np.any([(GT_img_v!= 85),(GT_img_v!= 170)], axis=0))
#
#         GT_img_v = GT_img_v==255
#
#         # conf_matrix[:,:,img] = confusion_matrix(GT_img_v[valid_pixels], Prediction_img_v[valid_pixels])
#         conf_matrix[:,:,img] = confusion_matrix(GT_img_v, Prediction_img_v)
#
#     return conf_matrix
#
# def getVectors(GT, Prediction):
#     if len(GT) != len(Prediction):
#         print "Datasets does not have same size"
#         return -1
#
#     img_shape = np.shape(Prediction[0])
#
#     gt_vector = np.array([],dtype=np.bool)
#     prediction_vector = np.array([],dtype=np.bool)
#
#     # sys.stdout.write("computing confusion matrix...")
#     # sys.stdout.flush()
#
#     for img in range(len(GT)):
#         GT_img = np.array(GT[img])
#         GT_img = GT_img[:,:,0]
#
#         Prediction_img = np.array(Prediction[img])
#
#         Prediction_img_v = Prediction_img.reshape((img_shape[0]*img_shape[1]))
#         GT_img_v = GT_img.reshape((img_shape[0]*img_shape[1]))
#
#         valid_pixels = np.nonzero(np.any([(GT_img_v!= 85),(GT_img_v!= 170)], axis=0))
#         GT_img_v = GT_img_v==255
#
#         gt_vector = np.append(gt_vector,GT_img_v[valid_pixels])
#         prediction_vector = np.append(prediction_vector,Prediction_img_v[valid_pixels])
#
#     return gt_vector, prediction_vector
#
# def Metrics(conf_matrix, show=True):
#     conf_matrix_total = np.sum(conf_matrix,axis=2)
#     TN = conf_matrix_total[0,0]
#     TP = conf_matrix_total[1,1]
#     FP = conf_matrix_total[0,1]
#     FN = conf_matrix_total[1,0]
#
#     if show:
#         print "Computing Metrics with TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
#     Accuracy = (float(TP) + float(TN)) / (float(TP) + float(TN) + float(FP) + float(FN))
#     Recall = float(TP) / (float(TP) + float(FN))
#
#     if ((float(TP) + float(FP)) > 0):
#         Precision = float(TP) / (float(TP) + float(FP))
#     else:
#         Precision = 0
#
#     if ((float(Precision) + float(Recall)) > 0):
#         F1 = 2 * float(Precision) * float(Recall) / (float(Precision) + float(Recall))
#     else:
#         F1 = 0
#
#     if show:
#         print "Accuracy: " + str(Accuracy * 100)
#         print "Recall: " + str(Recall * 100)
#         print "Precision: " + str(Precision * 100)
#         print "F1: " + str(F1 * 100)
#     return Accuracy, Recall, Precision, F1
#
# def ConfusionMatrix_own(GT, Prediction):
#     if len(GT) != len(Prediction):
#         print "Datasets does not have same size"
#         return -1
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#
#     TPFv = []
#     TPGTv = []
#     F1v = []
#
#     print "Computing Confusion Matrix"
#     for img in range(len(GT)):
#         GT_img = np.array(GT[img])
#         Prediction_img = np.array(Prediction[img])
#         print "Computing image: " + str(img)
#         TPF = 0
#         TNF = 0
#         FPF = 0
#         FNF = 0
#         TPFGT = 0
#         for i in range(GT_img.shape[0]):
#             for j in range(GT_img.shape[1]):
#                 GT_value = GT_img[i][j][0]
#                 Prediction_value = Prediction_img[i][j][0]
#                 if GT_value >= 170:
#                     GT_value = 1
#                     TPFGT += 1
#                 else:
#                     GT_value = 0
#                 if 1 == Prediction_value and GT_value == 1:
#                     TP += 1
#                     TPF += 1
#                 elif 0 == Prediction_value and GT_value == 0:
#                     TN += 1
#                     TNF += 1
#                 elif 1 == Prediction_value and GT_value == 0:
#                     FP += 1
#                     FPF += 1
#                 elif 0 == Prediction_value and GT_value == 1:
#                     FN += 1
#                     FNF += 1
#
#         AF, RF, PF, F1F = Metrics(TPF, TNF, FPF, FNF, False)
#         TPFv.append(TPF)
#         TPGTv.append(TPFGT)
#         F1v.append(F1F)
#     return [TP, TN, FP, FN], TPFv, TPGTv, F1v
#
# def Metrics_own(TP, TN, FP, FN, show=True):
#     if show:
#         print "Computing Metrics with TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
#     Accuracy = (float(TP) + float(TN)) / (float(TP) + float(TN) + float(FP) + float(FN))
#     Recall = float(TP) / (float(TP) + float(FN))
#
#     if ((float(TP) + float(FP)) > 0):
#         Precision = float(TP) / (float(TP) + float(FP))
#     else:
#         Precision = 0
#
#     if ((float(Precision) + float(Recall)) > 0):
#         F1 = 2 * float(Precision) * float(Recall) / (float(Precision) + float(Recall))
#     else:
#         F1 = 0
#
#     if show:
#         print "Accuracy: " + str(Accuracy * 100)
#         print "Recall: " + str(Recall * 100)
#         print "Precision: " + str(Precision * 100)
#         print "F1: " + str(F1 * 100)
#     return Accuracy, Recall, Precision, F1
#
# def ConfusionMatrixDesync(GT, Prediction):
#     if len(GT) != len(Prediction):
#         print "Datasets does not have same size"
#         return -1
#     vDelay = []
#     for delay in range(0,21,5):
#         TP = 0
#         TN = 0
#         FP = 0
#         FN = 0
#
#         TPFv = []
#         TPGTv = []
#         F1v = []
#
#         print "Computing Confusion Matrix Delay " + str(delay)
#         for img in range(len(GT)-delay):
#         # for img in range(10):
#             GT_img = np.array(GT[img])
#             Prediction_img = np.array(Prediction[img+delay])
#             print "Computing image: " + str(img)
#             TPF = 0
#             TNF = 0
#             FPF = 0
#             FNF = 0
#             TPFGT = 0
#             for i in range(GT_img.shape[0]):
#                 for j in range(GT_img.shape[1]):
#                     GT_value = GT_img[i][j][0]
#                     Prediction_value = Prediction_img[i][j][0]
#                     if GT_value >= 170:
#                         GT_value = 1
#                         TPFGT += 1
#                     else:
#                         GT_value = 0
#                     if 1 == Prediction_value and GT_value == 1:
#                         TP += 1
#                         TPF += 1
#                     elif 0 == Prediction_value and GT_value == 0:
#                         TN += 1
#                         TNF += 1
#                     elif 1 == Prediction_value and GT_value == 0:
#                         FP += 1
#                         FPF += 1
#                     elif 0 == Prediction_value and GT_value == 1:
#                         FN += 1
#                         FNF += 1
#
#             AF, RF, PF, F1F = Metrics(TPF, TNF, FPF, FNF, False)
#             TPFv.append(TPF)
#             TPGTv.append(TPFGT)
#             F1v.append(F1F)
#         vDelay.append([[TP, TN, FP, FN], TPFv, TPGTv, F1v])
#
#     return vDelay
#
# def showMultipleGraphic(x, y, label, labelx1, labelx2):
#     plt.plot(x[0], y[0], label=labelx1)
#     plt.plot(x[1], y[1], label=labelx2)
#     plt.xlabel("time")
#     plt.ylabel(label)
#     plt.show()
#
# def showSimpleGraphic(x, y, label):
#     plt.plot(x, y)
#     plt.xlabel("time")
#     plt.ylabel(label)
#     plt.show()
