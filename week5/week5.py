from objectTracker import *
from week5_utils import *
from dataset import Dataset
from computeVelocity import *

def main():

    # dataset_name = 'highway'
    dataset_name = 'traffic'

    if dataset_name == 'highway':
        frames_range = (1051, 1350)
        alpha = 1.9
        ro = 0.25
        p = 220
        conn = 4
        distThreshold = 40

    elif dataset_name == 'traffic':
        frames_range = (951, 1050)
        alpha = 2.8
        ro = 0.15
        p = 330
        conn = 4
        distThreshold = 40

    else:
        print "Invalid dataset name"
        return

    # Read dataset
    dataset = Dataset(dataset_name,frames_range[0], frames_range[1])

    imgs = dataset.readInput()
    imgs_GT = dataset.readGT()

    # Split dataset
    train = imgs[:len(imgs)/2]
    test = imgs[len(imgs)/2:]
    test_GT = imgs_GT[len(imgs)/2:]

    # Clean GT
    cleaned_GT = cleanGT(test_GT)
    # make_gif_mask(cleaned_GT)

    # Extract masks from sequences
    # results, metrics = mask_pipeline(train, test, test_GT, alpha, ro, conn, p, dataset_name, prints=True)

    # Perform the tracking
    images_bb, detections = objectTracker(test, cleaned_GT, distThreshold)

    # Compute velocities
    computeVelocity(test,detections)

    images_bb2 = drawBBoxes(test,detections)

    # make_gif(images_bb2, 'tracking.gif')

def drawBBoxes(images,detections):
    new_velocity_weight = 0.1
    for detection in detections:
        for i in range(1,len(detection.framesList)):
            mean_velocity = (detection.velocities[i])*new_velocity_weight + (detection.velocities[i-1])*(1-new_velocity_weight)
            images[detection.framesList[i]] = drawBBox(images[detection.framesList[i]], detection.bboxList[i], detection.id, vel=np.round(mean_velocity,decimals=2))
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

if __name__ == "__main__":
    main()
