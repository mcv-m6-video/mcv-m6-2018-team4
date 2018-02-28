from scipy import ndimage

from flow_utils import flow_read, flow_error_map
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.io as skimio
import os

# List of sequences to add to the measurement
sequence = '000045'  # '000157','000045'

# Read flow images
F_est = flow_read(os.path.join('results', 'data_stereo_flow', 'LKflow_' + sequence + '_10.png'))
F_gt = flow_read(os.path.join('..', '..', 'Datasets', 'data_stereo_flow', 'training', 'flow_noc', sequence + '_10.png'))
I = skimio.imread(os.path.join('..', '..', 'Datasets', 'data_stereo_flow', 'training', 'image_0', sequence + '_10.png'))

# Compute errors
(E, F_val) = flow_error_map(F_gt, F_est)

# Plot error image
plt.imshow(I, cmap='gray')
# plt.imshow(E,cmap='rainbow',alpha=.5)
plt.imshow(ndimage.grey_dilation(E, size=(3, 3)), cmap='rainbow', alpha=.5)
plt.tick_params(axis='both', labelbottom='off', labelleft='off')
plt.colorbar()
plt.savefig('error_im_' + sequence + '.png', dpi=300, bbox_inches='tight')

# Plot histogram
n, bins, patches = plt.hist(E[F_val == 1], bins=25)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
cm = plt.cm.get_cmap('rainbow')
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
plt.savefig('error_hist_' + sequence + '.png', dpi=300, bbox_inches='tight')
