import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class Dataset:
    def __init__(self, dataset_name,frame_init=1, frame_end=None):
        if dataset_name == 'highway':
            path = "../../Datasets/highway/"

        if dataset_name == 'traffic':
            path = "../../Datasets/traffic/"

        if dataset_name == 'fall':
            path = "../../Datasets/fall/"

        self.gt_path = path + "groundtruth/"
        self.input_path = path + "input/"

        self.frames_range = (frame_init-1,frame_end)

    def readInput(self):
        print "Reading Input"
        images = self.readImages(self.input_path,'jpg',self.frames_range)
        return images

    def readGT(self):
        print "Reading Groundtruth"
        images = self.readImages(self.gt_path,'png',self.frames_range)
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
            frame_ranges[1] = len(files)

        for file in range(frame_ranges[0],frame_ranges[1]):
            if files[file].endswith("." + extension):
                # print "loading image " + os.path.join(path, files[file])
                image = cv2.imread(os.path.join(path, files[file]))
                imgs.append(image)
        return imgs


