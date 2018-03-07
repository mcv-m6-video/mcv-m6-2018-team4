import numpy as np
import cv2
import os
import evaluation as ev
from dataset import Dataset
from matplotlib import animation
import matplotlib.pyplot as plt

# global variables
substractor = 'MOG2'    #MOG, MOG2, LSBP
history = 500;          #Default 500
varThreshold = 16        #16
bShadowDetection = True;

gif = False
showFrames = True
PR_curve = True

def main():
    # Read dataset
    #dataset = Dataset('highway',1050, 1350)
    dataset = Dataset('fall', 1461, 1560)
    #dataset = Dataset('traffic', 951, 1050)
    
    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    # cap = cv2.VideoCapture('source.mp4')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = createBackgroundSubtractor()

    substracted = []
    gif_frames = []

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
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        plt.show()

        print len(precision)
        print 'precision'
        print precision
        print 'recall'
        print recall
        print 'Area under the curve '
        print auc_val

    if gif is True:
        fig = plt.figure()
        anim = animation.ArtistAnimation(fig, gif_frames, interval=len(imgs), blit=True)
        anim.save('animation.gif', writer='imagemagick', fps=10)
        plt.show()

    cv2.destroyAllWindows()

def createBackgroundSubtractor():
    if substractor == 'MOG':
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()       #backgroundRatio, noiseSigma
    elif substractor == 'MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=bShadowDetection)
    elif substractor == 'LSBP':
        fgbg = cv2.bgsegm.BackgroundSubtractorLSBP()

    return fgbg


if __name__ == "__main__":
    main()
# #  GIF
# ims = []
# fig=plt.figure()
# for i in range(len(imgs)):
#     im = plt.imshow(imgs[i],cmap='gray',animated=True)
#     plt.axis("off")
#     ims.append([im])

# anim = animation.ArtistAnimation(fig, ims,interval=len(imgs), blit=True)
# anim.save('animation.gif', writer='imagemagick', fps=10)
# plt.show()

# while(1):
# # for img in imgs:
#     ret, frame = cap.read()
#     frame = img
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
    
#     if k == 27:
#         break

# cap.release()
