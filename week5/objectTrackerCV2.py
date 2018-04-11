import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

import cv2
import time

import sys

from DetectedObject import detectedObject


def objectTrackerCV2(images, masks, distThreshold, verbose=False):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if tracker_type == 'BOOSTING':
        tracker_creator = cv2.TrackerBoosting_create
    if tracker_type == 'MIL':
        tracker_creator = cv2.TrackerMIL_create
    if tracker_type == 'KCF':
        tracker_creator = cv2.TrackerKCF_create
    if tracker_type == 'TLD':
        tracker_creator = cv2.TrackerTLD_create
    if tracker_type == 'MEDIANFLOW':
        tracker_creator = cv2.TrackerMedianFlow_create
    if tracker_type == 'GOTURN':
        tracker_creator = cv2.TrackerGOTURN_create

    t = time.time()
    sys.stdout.write('Computing object tracking... ')

    frames_bb = []
    detections = []
    trackers = []
    car_counter = 0

    # First frame
    im_bb = images[0].copy()
    mask = masks[0].copy().astype(np.uint8)*255
    mask = np.tile(mask[:,:,np.newaxis],(1,1,3))
    #cv2.imshow('frame', mask)
    #cv2.waitKey(0)

    cc_mask = measure.label(mask, background=0) #get conected components
    nlbl = cc_mask.flatten().max()
    for lbl in range(1,nlbl+1):
        obj_coord = np.nonzero(cc_mask == lbl)
        car_counter += 1
        object = detectedObject(car_counter, obj_coord, 0)
        bbox_cv2 = bbox_our_to_cv2(object.bbox) #(object.bbox[2],object.bbox[0],object.bbox[1]-object.bbox[0], object.bbox[3]-object.bbox[2])
        tracker = tracker_creator()
        ok = tracker.init(mask, bbox_cv2) # Passing as input frame the current mask
        trackers.append(tracker)
        detections.append(object) #add object to the detections list
        im_bb = drawBBox(im_bb, object.bbox, object.id) #draw the bounding box in the image
        mask = drawBBox(mask, object.bbox, object.id)
    cv2.imshow('frame', mask)
    cv2.waitKey(0)

    frames_bb.append(im_bb)

    # Rest of frames
    for i in range(1,len(images)):
        im_bb = images[i].copy()
        mask = masks[i].copy().astype(np.uint8) * 255
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        oklist=[]
        for abc in range(len(trackers)):
            if not detections[abc].finished:
                ok, bbox = trackers[abc].update(mask)
                if not ok:
                    detections[abc].finished=True
                    continue
                our_bbox = bbox_cv2_to_our(bbox)
                detections[abc].updateBBox(our_bbox, i)
                im_bb = drawBBox(im_bb, detections[abc].bbox, detections[abc].id)  # draw the bounding box in the image
                mask = drawBBox(mask, detections[abc].bbox, detections[abc].id)

        sys.stdout.write('\n')
        cv2.imshow('frame', mask)
        cv2.waitKey(0)

        """
        Para cada connected component:
        1.- Comprobar IOU para ver si es el mismo
        2.- En tal caso hacer el update BBox (ahora si) basandonos en el bbox de cc
        3.- Hacemos init del algoritmo
        """
        cc_mask = measure.label(mask, background=0)  # get conected components
        nlbl = cc_mask.flatten().max()
        for lbl in range(1, nlbl + 1):
            # Create a temporal object to analize
            obj_coord = np.nonzero(cc_mask == lbl)
            object = detectedObject(np.nan, obj_coord, i)

        """
        for detection in detections:
            # Make all the detections not visible
            detection.visible = False

            # Make all the detections not visible
            if((i- detection.framesList[-1])>5 and not detection.finished):
                detection.finished = True
                if verbose: print "Object "+ str(detection.id)+ "deleted"

        cc_mask = measure.label(mask, background=0) #get conected components
        nlbl = cc_mask.flatten().max()
        for lbl in range(1, nlbl + 1):
            # Create a temporal object to analize
            obj_coord = np.nonzero(cc_mask == lbl)
            object = detectedObject(np.nan, obj_coord, i)

            # Search variables inizalitation
            minDist = np.inf
            objectFound = False

            print "Label "+ str(lbl)

            for detection in detections:
                if not detection.finished:
                    # Predict position with Kalman
                    prediction = detection.kalman.predict()
                    dist = euclideanDistance(object.center, prediction)

                    print "\t dist to " + str(detection.id)+ ": " +  str(dist)
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
                print "\n\t assigned to " + str(nearest_detection.id)
            else:
                # Add the new detected object to the detections list
                car_counter +=1
                object.id = car_counter
                detections.append(object)
                print "\n\t Created object " + str(nearest_detection.id)
        """

        # Draw the bounding boxes for the visible detections
        for detection in detections:
            if not detection.finished:
                im_bb = drawBBox(im_bb, detection.bbox, detection.id, center=(int(detection.center[1]),int(detection.center[0])), kalman=(int(detection.kalman.priorEstimateY), int(detection.kalman.priorEstimateX)))

        # cv2.imshow('frame', im_bb)
        # cv2.waitKey(0)

        frames_bb.append(im_bb)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')
    print "Cars detected " + str(car_counter)
    return frames_bb, detections

def drawBBox(im, bbox, id, vel=None, center=None, kalman=None):
    # bounding box
    topLeft = (bbox[2], bbox[0])
    bottomRight = (bbox[3], bbox[1])
    color = (0, 255, 0)
    border_size = 2
    im = cv2.rectangle(im, topLeft, bottomRight, color, border_size)

    # Rectangle labels
    labelHeight = 15
    padding = 2
    bottomHalf = bbox[2] + (bbox[3] - bbox[2]) / 2

    labtopLeft = (bbox[2], bbox[1] + padding)
    labbottomHRight = (bottomHalf, bbox[1] + labelHeight)

    labtopHleft = (bottomHalf, bbox[1] + padding)
    labbottomRight = (bbox[3], bbox[1] + labelHeight)

    # bottomHRight = (bottomLeft, map(lambda x: x/2, bottomRight))
    # topHLeft = (map(lambda x: x/2, bottomLeft), bottomRight)

    im = cv2.rectangle(im, labtopLeft, labbottomHRight, (96, 255, 96), -1)
    im = cv2.rectangle(im, labtopHleft, labbottomRight, (192, 255, 192), -1)

    # id
    # Center for the label
    xCenter = bbox[2] + padding*2
    yCenter = bbox[1] + padding*2 + labelHeight / 2
    idCenterPos = (xCenter, yCenter)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 0.5
    font_thickness = 1
    black_color = (0, 0, 0)
    text="ID: "+str(id)
    im = cv2.putText(im, text, idCenterPos, font, font_scale, black_color, font_thickness, cv2.LINE_AA)

    # vel
    # Center for the label
    xCenter = bottomHalf + padding
    idCenterPos = (xCenter, yCenter)
    if vel != None:
        text = str(np.round(vel,decimals=2)) + " km/h"
    else:
        text = ""
    im = cv2.putText(im, text, idCenterPos, font, font_scale, black_color, font_thickness, cv2.LINE_AA)

    blue_color = (255, 0, 0)
    # center bbox
    if center == None:
        im = cv2.circle(im, center, 4, blue_color, -1)

    # kalman
    if kalman == None:
        im = cv2.circle(im, kalman, 4, (0, 0, 255), -1)

    return im


def euclideanDistance(point1, point2):
    dist = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return dist

def bbox_cv2_to_our(bbox):
    return (int(bbox[1]), int(bbox[2] + bbox[1]), int(bbox[0]), int(bbox[3] + bbox[0]))

def bbox_our_to_cv2(bbox):
    return (int(bbox[2]), int(bbox[0]), int(bbox[1] - bbox[0]), int(bbox[3] - bbox[2]))


def bb_intersection_over_union(boxA, boxB):
    # Done for topleft bottright
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou