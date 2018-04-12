import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

import cv2
import time

import sys

from DetectedObject import detectedObject
#from week5 import drawBBox


def objectTrackerCV2(images, masks, verbose=False):
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
    # Get the frame and mask ready to use AND PLOT
    im_bb = images[0].copy()
    mask = masks[0].copy().astype(np.uint8) * 255
    mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
    # cv2.imshow('frame', mask)
    # cv2.waitKey(0)

    cc_mask = measure.label(mask, background=0)  # get conected components
    nlbl = cc_mask.flatten().max()
    for lbl in range(1, nlbl + 1):
        obj_coord = np.nonzero(cc_mask == lbl)
        object = detectedObject(car_counter, obj_coord, 0)
        car_counter += 1

        bbox_cv2 = bbox_our_to_cv2(
            object.bbox)  # (object.bbox[2],object.bbox[0],object.bbox[1]-object.bbox[0], object.bbox[3]-object.bbox[2])
        tracker = tracker_creator()
        ok = tracker.init(im_bb, bbox_cv2)  # Passing as input frame the current mask

        trackers.append(tracker)
        detections.append(object)  # add object to the detections list

        # im_bb = drawBBox(im_bb, object.bbox, object.id) #draw the bounding box in the image
        # mask = drawBBox(mask, object.bbox, object.id)

    # cv2.imshow('frame', mask)
    # cv2.waitKey(0)

    frames_bb.append(im_bb)

    # Rest of frames
    for i in range(1, len(images)):
        # Get the frame and mask ready to use AND PLOT
        im_bb = images[i].copy()
        mask = masks[i].copy().astype(np.uint8) * 255
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        # Get conected components
        cc_mask = measure.label(mask, background=0)
        nlbl = cc_mask.flatten().max()
        ccobjects = []
        for lbl in range(1, nlbl + 1):
            # Create a temporal object to analize
            obj_coord = np.nonzero(cc_mask == lbl)
            object = detectedObject(np.nan, obj_coord, i)
            ccobjects.append(object)

        # Track detections
        for j in range(len(trackers)):
            if not detections[j].finished:
                # Get tracked bbox
                ok, bbox = trackers[j].update(im_bb)

                # Check if the object still is visible
                if not ok:
                    detections[j].finished = True
                    continue
                our_bbox = bbox_cv2_to_our(bbox)  # convert bbox coords

                # Find for the corresponding ccl
                maxiou = -np.inf
                for cc in ccobjects:
                    iou = bbox_intersection_over_union(cc.bbox, our_bbox)
                    if iou > maxiou:
                        maxiou = iou
                        best_match = cc

                # Sometimes fails, so it needs an if
                if len(ccobjects) == 0:
                    detections[j].finished = True
                    continue
                # Delete from ccl list
                ccobjects.remove(best_match)

                # Update detection with the cc bbox and reinit tracker
                # (done because of the size and shape changes of the elements in the mask)
                # mask = drawBBox(mask, best_match.bbox, detections[j].id)
                detections[j].updateBBox(best_match.bbox, i)
                if False:  # maxiou < 1:
                    trackers[j] = tracker_creator()
                    trackers[j].init(mask, bbox_our_to_cv2(best_match.bbox))

        # Check for new objects
        for cc in ccobjects:
            cc.id = car_counter
            car_counter += 1
            bbox_cv2 = bbox_our_to_cv2(cc.bbox)
            tracker = tracker_creator()
            ok = tracker.init(im_bb, bbox_cv2)  # Passing as input frame the current mask
            trackers.append(tracker)
            detections.append(cc)

        # Draw bboxes
        # for detection in detections:
        #     if not detection.finished:
        #         im_bb = drawBBox(im_bb, detection.bbox, detection.id)  # draw the bounding box in the image
        #         mask = drawBBox(mask, detection.bbox, detection.id)

        # sys.stdout.write('{}:'.format(i))
        # for detection in detections:
        #     if not detection.finished:
        #         sys.stdout.write('{},'.format(detection.id))
        # sys.stdout.write('  len:{}\n'.format(len(detections)))
        # cv2.imshow('frame', mask)
        # cv2.waitKey(0)

        # cv2.imshow('frame', im_bb)
        # cv2.waitKey(0)

        # frames_bb.append(im_bb)

    elapsed = time.time() - t
    sys.stdout.write(str(elapsed) + ' sec \n')
    print "Cars detected " + str(car_counter)
    return detections


def euclideanDistance(point1, point2):
    dist = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return dist


def bbox_cv2_to_our(bbox):
    return (int(bbox[1]), int(bbox[2] + bbox[1] - 1), int(bbox[0]), int(bbox[3] + bbox[0] - 1))


def bbox_our_to_cv2(bbox):
    return (int(bbox[2]), int(bbox[0]), int(bbox[1] - bbox[0] + 1), int(bbox[3] - bbox[2] + 1))


def bbox_intersection_over_union(boxA, boxB):
    # Done for topleft bottright bbox with 0,0 in topleft corner of image
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
