import numpy as np
import cv2
from skimage import measure

from week5 import *
from DetectedObject import detectedObject

def objectTracker(images, masks, distThreshold):
    t = time.time()
    sys.stdout.write('Computing object tracking... ')

    frames_bb = []
    detections = []
    car_counter = 0

    # First frame
    im_bb = images[0].copy()
    mask = masks[0]

    cc_mask = measure.label(mask, background=0) #get conected components
    nlbl = cc_mask.flatten().max()
    for lbl in range(1,nlbl+1):
        obj_coord = np.nonzero(cc_mask == lbl)
        car_counter += 1
        object = detectedObject(car_counter, obj_coord, 0)
        detections.append(object) #add object to the detections list
        im_bb = drawBBox(im_bb, object.bbox, object.id) #draw the bounding box in the image

    frames_bb.append(im_bb)

    # Rest of frames
    for i in range(1,len(images)):
        im_bb = images[i].copy()
        mask = masks[i]

        # Make all the detections not visible
        for detection in detections:
            detection.visible = False

        cc_mask = measure.label(mask, background=0) #get conected components
        nlbl = cc_mask.flatten().max()
        for lbl in range(1, nlbl + 1):
            # Create a temporal object to analize
            obj_coord = np.nonzero(cc_mask == lbl)
            object = detectedObject(np.nan, obj_coord, i)

            # Search variables inizalitation
            minDist = np.inf
            objectFound = False

            for detection in detections:

                # Predict position with Kalman
                prediction = detection.kalman.predict()
                dist = euclideanDistance(object.center, prediction)

                # Search the nearest object
                if(dist < distThreshold):
                    if dist < minDist:
                        minDist = dist
                        nearest_detection = detection
                        objectFound = True

            if objectFound:
                # Update Kalman
                nearest_detection.kalman.update(object.center)

                # Make detection visible and update the Bounding Box
                nearest_detection.visible = True
                nearest_detection.updateBBox(object.bbox, i)
            else:
                # Add the new detected object to the detections list
                car_counter +=1
                object.id = car_counter
                detections.append(object)

        # Draw the bounding boxes for the visible detections
        for detection in detections:
            if detection.visible:
                im_bb = drawBBox(im_bb, detection.bbox, detection.id)

        frames_bb.append(im_bb)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')
    print "Cars detected " + str(car_counter)
    return frames_bb, detections

def drawBBox(im, bbox, id):
    topLeft = (bbox[2], bbox[0])
    bottomRight = (bbox[3], bbox[1])
    color = (0, 255, 0)
    border_size = 2
    im = cv2.rectangle(im, topLeft, bottomRight, color, border_size)

    bottomLeft = (bbox[2], bbox[1]+5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    im = cv2.putText(im, str(id), bottomLeft, font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)

    return im


def euclideanDistance(point1, point2):
    dist = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return dist
