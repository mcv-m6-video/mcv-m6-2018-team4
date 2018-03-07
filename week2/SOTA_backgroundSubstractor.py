import numpy as np
import cv2
import os
import evaluation as ev
from dataset import Dataset
from matplotlib import animation
import matplotlib.pyplot as plt

# global variables
substractor = 'MOG'    #MOG, MOG2, GMG, LSBP
history = 500;          #Default 500
varThreshold = 16        #16
bShadowDetection = True;

gif = True
showFrames = True
PR_curve = False
Metrics = False

def main():
    # Read dataset
    #dataset = Dataset('highway',1050, 1350)
    #dataset = Dataset('fall', 1461, 1560)
    dataset = Dataset('traffic', 951, 1050)
    
    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    # cap = cv2.VideoCapture('source.mp4')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = createBackgroundSubtractor()

    substracted = []
    gif_frames = []
    fig = plt.figure()
    
    for i in range(len(imgs)):
        frame = imgs[i]
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        substracted.append(fgmask)

        if gif is True:
            g = plt.imshow(fgmask, cmap='gray', animated=True)
            plt.axis("off")
            gif_frames.append([g])

        if showFrames is True:
            cv2.imshow('input', imgs[i])
            cv2.imshow('frame', fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    if PR_curve is True:
        precision, recall, auc_val = ev.getPR_AUC(imgs_GT, substracted)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='g')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title("Precision-Recall curve - Fall - MOG2")
        plt.show()

        print len(precision)
        print 'precision'
        print sum(precision)/len(precision)
        print 'recall'
        print recall
        print 'Area under the curve '
        print auc_val

    if Metrics is True:
        metrics = ev.getMetrics(imgs_GT, substracted)
        print metrics

    if gif is True:
        anim = animation.ArtistAnimation(fig, gif_frames, interval=len(imgs), blit=True)
        anim.save('animation.gif', writer='imagemagick', fps=10)
        # plt.show()

    #cap.release()
    cv2.destroyAllWindows()

def createBackgroundSubtractor():
    if substractor == 'MOG':
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()       #backgroundRatio, noiseSigma
    elif substractor == 'MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=bShadowDetection)
    elif substractor == 'LSBP':
        fgbg = cv2.bgsegm.BackgroundSubtractorLSBP()
    elif substractor == 'GMG':
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    return fgbg


if __name__ == "__main__":
    main()
