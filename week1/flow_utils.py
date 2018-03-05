import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def flow_read(filename, sequence=None, plot_flow=False):
    # loads flow field F from png file

    # Read the image
    # -1 because the values are not limited to 255
    # OpenCV reads in BGR order of channels
    I = cv2.imread(filename, -1)

    # Representation:
    #   Vector of flow (u,v), plot_flow=False, 
    #   Boolean if pixel has a valid value (is not an occlusion)
    F_u = (I[:, :, 2] - 2. ** 15) / 64
    F_v = (I[:, :, 1] - 2. ** 15) / 64
    F_valid = I[:, :, 0]

    if plot_flow:
        optical_flow_plot(F_u, F_v, filename, sequence)

    # Matrix with vector (u,v) in the channel 0 and 1 and boolean valid in channel 2
    return np.transpose(np.array([F_u, F_v, F_valid]), axes=[1, 2, 0])


def flow_error_map(F_gt, F_est):
    # Remember: Flow vector = (u,v)

    # Compute error
    E_du = F_gt[:, :, 0] - F_est[:, :, 0]
    E_dv = F_gt[:, :, 1] - F_est[:, :, 1]
    E = np.sqrt(E_du ** 2 + E_dv ** 2)

    # Set the error of the non valid (occluded) pixels to 0
    F_gt_val = F_gt[:, :, 2]
    E[F_gt_val == 0] = 0

    return (E, F_gt_val)

def optical_flow_plot(u, v, filename, sequence):
    print 'filename'
    print filename
    I = cv2.imread(filename, -1)
    a, b, c = I.shape

    # F_u = u[0::10, 0::10]
    # F_v = v[0::10, 0::10]

    # F_u = np.reshape(u, (a,b))
    # F_v = np.reshape(v, (a,b))
    x,y = np.meshgrid(np.arange(0,b,1), np.arange(0,a,1))
    plt.imshow(mpimg.imread(os.path.join('..', 'Datasets', 'data_stereo_flow',
                                        'training', 'image_0', sequence + '_10.png')), cmap='gray')

    img = cv2.imread(os.path.join('..', 'Datasets', 'data_stereo_flow',
                                        'training', 'image_0', sequence + '_10.png'))

    plt.quiver(x[::10, ::10], y[::10, ::10], u[::10,::10], v[::10,::10], color='red', pivot='mid')
    plt.show()