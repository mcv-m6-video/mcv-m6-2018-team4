import cv2
import numpy as np
from matplotlib.patches import Rectangle
from skimage.util import pad
import matplotlib.pyplot as plt


def ssd(array1, array2):
    return np.sum(np.power(array1.astype(np.float32) - array2.astype(np.float32), 2))

def reorder_towards_center(vector, center):
    npvector = np.array(vector)
    distances = np.abs(npvector-center)
    return npvector[np.argsort(distances)]

def block_matching(im1, im2, block_size=(3, 3), area=(2 * 3 + 3, 2 * 3 + 3), error_func=ssd, error_thresh=1):
    debug = False
    if im1.shape != im2.shape:
        print('ERROR: Image shapes are not the same!')
        exit(-1)

    if len(im1.shape) == 2:
        im1 = np.copy(im1)[:, :, np.newaxis]
        im2 = np.copy(im2)[:, :, np.newaxis]
    rows, cols, channels = im1.shape

    odd_block = (block_size[0] % 2, block_size[1] % 2)
    halfs_block = (block_size[0] / 2, block_size[1] / 2)
    padding = (halfs_block[0], halfs_block[1])

    odd_area = (area[0] % 2, area[1] % 2)
    halfs_area = (area[0] / 2, area[1] / 2)

    im1 = cv2.copyMakeBorder(im1, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_REFLECT)
    im2 = cv2.copyMakeBorder(im2, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_REFLECT)
    result = np.empty([rows, cols, 2])  # step

    # IM1's double loop
    result_i = 0
    for i in range(padding[0], rows + padding[0]):
        result_j = 0
        for j in range(padding[1], cols + padding[1]):
            block1 = im1[i - halfs_block[0]:i + halfs_block[0] + odd_block[0],
                     j - halfs_block[1]:j + halfs_block[1] + odd_block[1], :]
            if debug:
                f1 = plt.figure()
                plt.imshow(block1)
                plt.title('block1 in ({},{})'.format(i, j))
                plt.show()

            area_range = ((i - halfs_area[0] if i - halfs_area[0] > padding[0] else padding[0],
                           i + halfs_area[0] + odd_area[0] if i + halfs_area[0] + odd_area[0] < rows + padding[0] \
                               else rows + padding[0]),
                          (j - halfs_area[1] if j - halfs_area[1] > padding[1] else padding[1],
                           j + halfs_area[1] + odd_area[1] if j + halfs_area[1] + odd_area[1] < cols + padding[1] \
                               else cols + padding[1]))

            if debug:
                f, ax = plt.subplots()
                ax.imshow(im1, alpha=.5)
                ax.imshow(im2, alpha=.5)
                ax.add_patch(Rectangle((area_range[1][0] - .5, area_range[0][0] - .5),
                                       area_range[1][1] - area_range[1][0],
                                       area_range[0][1] - area_range[0][0],
                                       linewidth=1,
                                       edgecolor='w',
                                       facecolor='none'))
                plt.show()

            block2 = im2[i - halfs_block[0]:i + halfs_block[0] + odd_block[0],
                     j - halfs_block[1]:j + halfs_block[1] + odd_block[1], :]
            no_flow_error = error_func(block1, block2)
            min_error = no_flow_error
            max_error = min_error

            # IM2's double loop
            k_vector = reorder_towards_center(range(area_range[0][0], area_range[0][1]),i)
            l_vector = reorder_towards_center(range(area_range[1][0], area_range[1][1]),j)
            for k in k_vector:
                for l in l_vector:
                    if k==i and j==l:
                        continue

                    block2 = im2[k - halfs_block[0]:k + halfs_block[0] + odd_block[0],
                             l - halfs_block[1]:l + halfs_block[1] + odd_block[1], :]
                    if debug:
                        f2 = plt.figure()
                        plt.imshow(block2)
                        plt.title('block2 in ({},{})'.format(k, l))
                        plt.show()
                    if debug:
                        f12 = plt.figure()
                        plt.imshow(block1, alpha=.5)
                        plt.imshow(block2, alpha=.5)
                        plt.title('block1 in ({},{}) block2 in ({},{})'.format(i, j, k, l))
                        plt.show()

                    cur_error = error_func(block1, block2)
                    if debug:
                        print('cur_error: {}'.format(cur_error))
                        plt.close(f2)
                        plt.close(f12)
                    if cur_error < min_error:
                        min_error = cur_error
                        move = (k - i, l - j)
                    if cur_error > max_error:
                        max_error = cur_error
            if debug:
                plt.close(f1)
            if np.abs(min_error - no_flow_error) < error_thresh:
                move = (0, 0)
            result[result_i, result_j, 0] = move[0]
            result[result_i, result_j, 1] = move[1]

            result_j += 1
        result_i += 1
    return result
# step step=(1, 1),
# same size ouput: np repeat
# if max-min error very small then no flow
