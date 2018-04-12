from objectTracker import *
from objectTrackerCV2 import objectTrackerCV2
from week5_utils import *
from dataset import Dataset
from computeVelocity import *
# from computeVelocity_own_H import *

def main():

    # dataset_name = 'highway'
    # dataset_name = 'traffic'
    # dataset_name = 'traffic_stab'
    dataset_name = 'week5_dataset'

    # dataset_name = 'sequence_parc_nova_icaria'

    read_dataset = True

    if dataset_name == 'highway':
        image_format = 'jpg'
        frames_range = (1051, 1350)
        alpha = 1.9
        ro = 0.25
        p = 220
        conn = 4
        distThreshold = 40
        ROI = None
        valid_pixels = None

    elif dataset_name == 'traffic':
        image_format = 'jpg'
        frames_range = (951, 1050)
        alpha = 2.8
        ro = 0.15
        p = 330
        conn = 4
        distThreshold = 40
        ROI = None
        valid_pixels = None

    elif dataset_name == 'traffic_stab':
        image_format = 'jpg'
        frames_range = (951, 1050)
        alpha = 3.5
        ro = 0.15
        p = 400
        conn = 4
        distThreshold = 40
        ROI = None

        seq_stab_np = np.load('../week4/video_stabilization/seq_stab.npy')
        GT_stab_np = np.load('../week4/video_stabilization/GTstab.npy')

        valid_frames_np = get_valid_mask(seq_stab_np)

        seq_stab = []
        GT_stab = []
        valid_frames = []
        for i in range(seq_stab_np.shape[3]):
            seq_stab.append(seq_stab_np[:, :, :, i].astype(np.uint8))
            GT_stab.append(GT_stab_np[:, :, :, i].astype(np.uint8))
            valid_frames.append(valid_frames_np[:, :, i])

        # Split dataset
        train = seq_stab[:len(seq_stab) / 2]
        test = seq_stab[len(seq_stab) / 2:]
        valid_pixels = valid_frames[len(seq_stab) / 2:]
        test_GT = GT_stab[len(seq_stab) / 2:]

        read_dataset = False

    # elif dataset_name == 'sequence_parc_nova_icaria':
    #     image_format = 'jpg'
    #     frames_range = (210, 610)
    #     alpha = 2.5
    #     ro = 0.15
    #     p = 300
    #     conn = 4
    #     distThreshold = 40

    elif dataset_name == 'week5_dataset':
        image_format = 'png'
        frames_range = (1, None)
        alpha = 2.5
        ro = 0.2
        p = 250
        conn = 4
        distThreshold = 40
        valid_pixels = None
        ROI = True

    else:
        print "Invalid dataset name"
        return

    if read_dataset:
        # Read dataset
        dataset = Dataset(dataset_name,frames_range[0], frames_range[1])
        ROI = dataset.getROI()

        imgs = dataset.readInput(image_format)
        # imgs_GT = dataset.readGT()

        # Split dataset
        train = imgs[:len(imgs)/2]
        test = imgs[len(imgs)/2:]
        # test_GT = imgs_GT[len(imgs)/2:]
        test_GT = []

    # Clean GT
    # cleaned_GT = cleanGT(test_GT)
    # make_gif_mask(cleaned_GT)

    # Extract masks from sequences
    results, metrics = mask_pipeline(train, test, test_GT, alpha, ro, conn, p, dataset_name, prints=True, ROI=ROI, valid_pixels=valid_pixels)

    # make_gif_mask(results)

    # Perform the tracking
    images_bb, detections = objectTracker(test, results, distThreshold)
    #detections = objectTrackerCV2(test, results)

    # Compute velocities
    if dataset_name == 'week5_dataset':
        computeVelocity_own(test,detections)
    else:
        computeVelocity(test,detections)

    images_bb2 = drawBBoxes(test,detections)
    #
    make_gif(images_bb2, 'tracking.gif')

def drawBBoxes(images,detections):
    new_velocity_weight = 0.3
    for detection in detections:
        mean_velocities = []
        for i in range(1,len(detection.framesList)):
            mean_velocity = (detection.velocities[i])*new_velocity_weight + (detection.velocities[i-1])*(1-new_velocity_weight)
            mean_velocity = mean_velocity+ 60
            images[detection.framesList[i]] = drawBBox(images[detection.framesList[i]], detection.bboxList[i], detection.id, vel=np.round(mean_velocity,decimals=2),limit=mean_velocity>80)
            mean_velocities.append(mean_velocity)
        print "Car " + str(detection.id) + ": " + str(np.mean(np.array(mean_velocities)))

    for i in range(len(images)):
        cv2.imshow("frame",images[i])
        cv2.waitKey(0)

    return images


def cleanGT(GT):
    cleaned_GT = []
    for i in range(len(GT)):
        gt_i = GT[i][:,:,0]
        gt_true = (gt_i >= 170)

        cleaned_GT.append(gt_true)

    return cleaned_GT

def get_valid_mask(sequence):

    valid_frames = np.zeros((sequence.shape[0],sequence.shape[1],sequence.shape[3]),dtype=np.bool)
    for i in range(sequence.shape[3]):
        frame = sequence[:,:,:,i]
        frame = (frame == 0)
        valid_frame = np.logical_and(frame[:, :, 0], frame[:, :, 1], frame[:, :, 2])
        valid_frames[:,:,i] = valid_frame

    return valid_frames


def drawBBox(im, bbox, id, vel=None, center=None, kalman=None, limit=False):
    # bounding box
    topLeft = (bbox[2], bbox[0])
    bottomRight = (bbox[3], bbox[1])
    if not limit:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    border_size = 2
    im = cv2.rectangle(im, topLeft, bottomRight, color, border_size)

    # # Rectangle labels
    # labelHeight = 15
    # padding = 2
    # bottomHalf = bbox[2] + (bbox[3] - bbox[2]) / 2
    #
    # labtopLeft = (bbox[2], bbox[1] + padding)
    # labbottomHRight = (bottomHalf, bbox[1] + labelHeight)
    #
    # labtopHleft = (bottomHalf, bbox[1] + padding)
    # labbottomRight = (bbox[3], bbox[1] + labelHeight)
    #
    # im = cv2.rectangle(im, labtopLeft, labbottomHRight, (96, 255, 96), -1)
    # im = cv2.rectangle(im, labtopHleft, labbottomRight, (192, 255, 192), -1)

    # Rectangle labels (fix size)
    labelHeight = 15
    labelWidth = 30;
    labelWidth2 = 50;
    padding = 2
    bottomHalf = bbox[2] + labelWidth

    labtopLeft = (bbox[2], bbox[1] + padding)
    labbottomHRight = (bottomHalf, bbox[1] + labelHeight)

    labtopHleft = (bottomHalf, bbox[1] + padding)
    labbottomRight = (bottomHalf + labelWidth2, bbox[1] + labelHeight)

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

if __name__ == "__main__":
    main()
