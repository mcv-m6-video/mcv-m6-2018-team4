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

        for detection in detections:
            # Make all the detections not visible
            detection.visible = False

            # Make all the detections not visible
            if((i- detection.framesList[-1])>5):
                detection.finished = True

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
                if not detection.finished:
                    # Predict position with Kalman
                    prediction = detection.kalman.predict()
                    dist = euclideanDistance(object.center, prediction)

                    print "dist to  " + str(detection.id)+ ": " +  str(dist)
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
            if not detection.finished:
                im_bb = drawBBox(im_bb, detection.bbox, detection.id, center=(int(detection.center[1]),int(detection.center[0])), kalman=(int(detection.kalman.priorEstimateY), int(detection.kalman.priorEstimateX)))

        cv2.imshow('frame', im_bb)
        cv2.waitKey(0)

        frames_bb.append(im_bb)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')
    print "Cars detected " + str(car_counter)
    return frames_bb, detections

def drawBBox(im, bbox, id, center=(0,0), kalman=(0,0)):
    # bounding box
    topLeft = (bbox[2], bbox[0])
    bottomRight = (bbox[3], bbox[1])
    color = (0, 255, 0)
    border_size = 2
    im = cv2.rectangle(im, topLeft, bottomRight, color, border_size)

    # id
    bottomLeft = (bbox[2], bbox[1]+5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    blue_color = (255, 0, 0)
    im = cv2.putText(im, str(id), bottomLeft, font, font_scale, blue_color, font_thickness, cv2.LINE_AA)

    # center bbox
    im = cv2.circle(im, center, 4, blue_color, -1)

    # kalman
    im = cv2.circle(im, kalman, 4,  (0, 0, 255), -1)

    return im


def euclideanDistance(point1, point2):
    dist = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return dist
