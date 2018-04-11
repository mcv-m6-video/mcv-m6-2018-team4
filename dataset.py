import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
import time


class Dataset:
    def __init__(self, dataset_name,frame_init=1, frame_end=None):
        if dataset_name == 'highway':
            path = "../../Datasets/highway/"

        elif dataset_name == 'traffic':
            path = "../../Datasets/traffic/"

        elif dataset_name == 'fall':
            path = "../../Datasets/fall/"

        elif dataset_name == 'sequence_parc_nova_icaria':
            path = "../../Datasets/sequence_parc_nova_icaria/"

        elif dataset_name == 'week5_dataset':
            path = "../../Datasets/week5_dataset/"
        else:
            print "Invalid dataset name"
            return


        self.gt_path = path + "groundtruth/"
        self.input_path = path + "input/"

        if dataset_name == 'week5_dataset':
            self.ROI = ROI = cv2.imread(path+'ROI.jpg')
            self.ROI = [ROI[:,:,0]<150]
        else:
            self.ROI = None

        self.frames_range = (frame_init-1,frame_end)

    def readInput(self,format):
        t = time.time()
        sys.stdout.write('Reading Input... ')

        images = self.readImages(self.input_path, format, self.frames_range)
        self.input = images
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

        return images

    def readGT(self):
        t = time.time()
        sys.stdout.write('Reading Groundtruth... ')

        images = self.readImages(self.gt_path,'png',self.frames_range)
        self.gt = images
        elapsed = time.time() - t
        sys.stdout.write(str(elapsed) + ' sec \n')

        return images

    def readImages(self, path, extension, frame_ranges):
        if extension == "" or (extension != "jpg" and extension != "png") or not os.path.exists(path):
            print "Extension or path is invalid"
            return -1
        if len(os.listdir(path)) == 0:
            print "Directory is empty"
            return 0

        imgs = []
        files = sorted(os.listdir(path))
        if frame_ranges[1] == None:
            frame_ranges = (frame_ranges[0], len(files))

        for file in range(frame_ranges[0],frame_ranges[1]):
            if files[file].endswith("." + extension):
                # print "loading image " + os.path.join(path, files[file])
                image = cv2.imread(os.path.join(path, files[file]))
                imgs.append(image)
        return imgs

    def get_numpy_gray(self, sequence):
        h,w,c = sequence[0].shape
        gray_sequence = np.zeros((h,w,len(sequence)))
        for i in range(len(sequence)):
            gray_image = cv2.cvtColor(sequence[i], cv2.COLOR_BGR2GRAY)
            gray_sequence[:,:,i] = gray_image

        return gray_sequence

    def getROI(self):
        return self.ROI



