import numpy as np
import cv2
from skimage import measure

from week5 import *
from DetectedObject import detectedObject

def objectTracker(images, masks):
    t = time.time()
    sys.stdout.write('Computing object tracking... ')

    frames_bb = []
    detections = []
    car_counter = 0

    # First frame
    im_bb = images[0]
    mask = masks[0]
    cc_mask = measure.label(mask, background=0)
    nlbl = cc_mask.flatten().max()
    for lbl in range(1,nlbl+1):
        obj_coord = np.nonzero(cc_mask == lbl)
        object = detectedObject(car_counter, obj_coord)
        detections.append(object)
        car_counter += 1
        im_bb = drawBBox(im_bb, object.bbox)

    # plt.imshow(im_bb)
    # plt.show()
    frames_bb.append(im_bb)

    for i in range(1,len(images)):
        im_bb = images[i]
        mask = masks[i]

        for detection in detections:
            detection.visible = False

        cc_mask = measure.label(mask, background=0)
        nlbl = cc_mask.flatten().max()
        for lbl in range(1, nlbl + 1):
            obj_coord = np.nonzero(cc_mask == lbl)
            object = detectedObject(np.nan, obj_coord)

            minDist = np.inf
            objectFound = False

            for detection in detections:

                prediction = detection.kalman.predict()
                dist = euclideanDistance(object.center, prediction)

                # if(distance < kalmanThreshold):
                if(dist < 80):
                    # Search the nearly object
                    if dist < minDist:
                        minDist = dist
                        nearest_detection = detection
                        objectFound = True

            if objectFound:
                nearest_detection.kalman.update(object.center)
                nearest_detection.updateBBox(object.bbox)
                nearest_detection.visible = True
            else:
                car_counter +=1
                object.id = car_counter
                detections.append(object)


        for detection in detections:
            if detection.visible:
                im_bb = drawBBox(im_bb, detection.bbox)

        # plt.imshow(im_bb)
        # plt.show()

        frames_bb.append(im_bb)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')

    return frames_bb

def drawBBox(im, bbox):
    topLeft = (bbox[2], bbox[0])
    bottomRight = (bbox[3], bbox[1])
    bottomLeft = (bbox[2], bbox[1])
    color = (0, 255, 0)
    border_size = 2
    im = cv2.rectangle(im, topLeft, bottomRight, color, border_size)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.2
    # font_thickness = 1
    # im = cv2.putText(im, 'ID', bottomLeft, font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)

    return im


def euclideanDistance(point1, point2):
    dist = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return dist
