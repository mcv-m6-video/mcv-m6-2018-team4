import cv2
import functions as fc
import numpy as np
import os
import time
from dataset import Dataset

def main():

    dataset = Dataset('highway',1,20)
    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    # ori_imgs = fc.readImages("../../Datasets/highway/groundtruth", "png")
    # A_imgs = fc.readImages("results/highway/A", "png")
    # B_imgs = fc.readImages("results/highway/B", "png")


    # TASK 1.1 - Background Substraction Evaluation

    # test A
    t = time.time()
    A_CMatrix = fc.ConfusionMatrix(ori_imgs, A_imgs)
    A_Metrics = fc.Metrics(A_CMatrix)
    elapsed = time.time() - t
    print('Elapsed time is ' + str(elapsed) + ' seconds')

    # test B
    t = time.time()
    B_CMatrix = fc.ConfusionMatrix_2(ori_imgs, B_imgs)
    B_Metrics = fc.Metrics_2(B_CMatrix)
    elapsed = time.time() - t
    print('Elapsed time is ' + str(elapsed) + ' seconds')


    # TASK 2.1 - Plot TP vs Time
    A_CMatrix, TPFv_A, TPGTv_A, F1v_A = fc.ConfusionMatrix(ori_imgs, A_imgs)
    B_CMatrix, TPFv_B, TPGTv_B, F1v_B = fc.ConfusionMatrix(ori_imgs, B_imgs)

    x = [np.arange(0, 200, 1), np.arange(0, 200, 1), np.arange(0, 200, 1)]
    y = [np.array(TPFv_A), np.array(TPFv_B), np.array(TPGTv_A)]
    axis = ["Time", "#Pixels"]
    labels = ["True Positives A", "True Positives B", "Total Positives"]
    fc.plotGraphics(x, y, axis, labels)

    # TASK 2.2 - Plot F1-score vs Time
    x = [np.arange(0, 200, 1), np.arange(0, 200, 1)]
    y = [np.array(F1v_A), np.array(F1v_B)]
    axis = ["Time", "F1-score"]
    labels = ["Test A", "Test B"]
    fc.plotGraphics(x, y, axis, labels)

    # TASK 4 - De-synchronization of results
    vDelay_A = fc.ConfusionMatrixDesync(ori_imgs, A_imgs)
    vDelay_B = fc.ConfusionMatrixDesync(ori_imgs, B_imgs)

    x = []
    yA = []
    yB = []
    labels = []

    for delay in range(len(vDelay_A)):
        [A_CMatrix, TPFv_A, TPGTv_A, F1v_A] = vDelay_A[delay]
        [B_CMatrix, TPFv_B, TPGTv_B, F1v_B] = vDelay_B[delay]
        labels.append( "Delay " + str(delay*5))
        x.append(np.arange(0, len(F1v_A), 1))
        yA.append(np.array(F1v_A))
        yB.append(np.array(F1v_B))

    axis = ["Time", "F1-score"]
    fc.plotGraphics(x, yA, axis, labels)
    fc.plotGraphics(x, yB, axis, labels)

if __name__ == "__main__":
    main()
