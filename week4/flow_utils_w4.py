import numpy as np
import cv2
from scipy import ndimage


def flow_visualization(u, v, dil_size=0):
    H = u.shape[0]
    W = u.shape[1]
    hsv = np.zeros((H, W, 3))

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(u, v)

    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # hsv[:,:,2] = ((mag - np.min(mag))/np.max(mag))*255
    hsv[:, :, 1] = 255

    hsv[:, :, 0] = ndimage.grey_dilation(hsv[:, :, 0], size=(dil_size, dil_size))
    hsv[:, :, 2] = ndimage.grey_dilation(hsv[:, :, 2], size=(dil_size, dil_size))

    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype=np.uint8)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow


def flow_read(filename):
    # loads flow field F from png file

    # Read the image
    # -1 because the values are not limited to 255
    # OpenCV reads in BGR order of channels
    I = cv2.imread(filename, -1)

    # Representation:
    #   Vector of flow (u,v)
    #   Boolean if pixel has a valid value (is not an occlusion)
    F_u = (I[:, :, 2] - 2. ** 15) / 64
    F_v = (I[:, :, 1] - 2. ** 15) / 64
    F_valid = I[:, :, 0]

    # Matrix with vector (u,v) in the channel 0 and 1 and boolean valid in channel 2
    return np.transpose(np.array([F_u, F_v, F_valid]), axes=[1, 2, 0])

def flo_flow_read(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file: {}'.format(filename)
        else:
            h = np.fromfile(f, np.int32, count=1)
            w = np.fromfile(f, np.int32, count=1)
            print 'Reading %d x %d flo file' % (w[0], h[0])
            data = np.fromfile(f, np.float32, count=2 * h * w)
            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (w[0], h[0], 2))

def flow_error(F_gt, F_est):
    # Remember: Flow vector = (u,v)

    # Compute error
    E_du = F_gt[:, :, 0] - F_est[:, :, 0]
    E_dv = F_gt[:, :, 1] - F_est[:, :, 1]
    E = np.sqrt(E_du ** 2 + E_dv ** 2)

    # Set the error of the non valid (occluded) pixels to 0
    F_gt_val = F_gt[:, :, 2]
    E[F_gt_val == 0] = 0

    MSE = np.mean(E[F_gt_val != 0])
    PEPN = np.sum(E[F_gt_val != 0] > 3) * 100. / len(E[F_gt_val != 0])

    print('MSE: ' + str(MSE))
    print('PEPN: ' + str(PEPN))

    return MSE, PEPN