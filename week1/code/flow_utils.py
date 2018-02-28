import cv2
import numpy as np


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


def flow_error_image(F_gt, F_est, tau, dilate_radius=1):
    (E, F_val) = flow_error_map(F_gt, F_est)
    F_mag = np.sqrt(F_gt[:, :, 0] ** 2 + F_gt[:, :, 1] ** 2)
    aux = E / F_mag
    aux[np.isnan(aux)] = np.inf
    E_im = np.minimum(E / tau[0], aux / tau[1])
    return E_im / np.max(E_im) * 255

    ''' MATLAB Code
    [E,F_val] = flow_error_map (F_gt,F_est);
    F_mag = sqrt(F_gt(:,:,1).*F_gt(:,:,1)+F_gt(:,:,2).*F_gt(:,:,2));
    E = min(E/tau(1),(E./F_mag)/tau(2));
    
    cols = error_colormap();
    
    F_err = zeros(size(F_gt));
    for i=1:size(cols,1)
      [v,u] = find(F_val > 0 & E >= cols(i,1) & E <= cols(i,2));
      F_err(sub2ind(size(F_err),v,u,1*ones(length(v),1))) = cols(i,3);
      F_err(sub2ind(size(F_err),v,u,2*ones(length(v),1))) = cols(i,4);
      F_err(sub2ind(size(F_err),v,u,3*ones(length(v),1))) = cols(i,5);
    end
    
    F_err = imdilate(F_err,strel('disk',dilate_radius));
    '''
