from objectTracker import *
from week5_utils import *
from dataset import Dataset

def main():

    dataset_name = 'highway'
    # dataset_name = 'traffic'

    if dataset_name == 'highway':
        frames_range = (1051, 1350)
        alpha = 1.9
        ro = 0.25
        p = 220
        conn = 4
        distThreshold = 30

    elif dataset_name == 'traffic':
        frames_range = (951, 1050)
        alpha = 2.8
        ro = 0.15
        p = 330
        conn = 4
        distThreshold = 80

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

    # Extract masks from sequences
    if dataset_name == 'highway':
        results, metrics = morphology_highway(train, test, test_GT, alpha, ro, conn, p, prints=True, valid_pixels=None)

    elif dataset_name == 'traffic':
        results, metrics = morphology_traffic(train, test, test_GT, alpha, ro, conn, p, prints=True, valid_pixels=None)
    else:
        print "Invalid dataset name"
        return

    # Perform the tracking
    tracking = objectTracker(test, results, distThreshold)

    make_gif(tracking, 'tracking.gif')


if __name__ == "__main__":
    main()
