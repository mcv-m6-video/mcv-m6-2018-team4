from flow_utils import flow_read, flow_error_map
import numpy as np
import os

# List of sequences to add to the measurement
sequences = ['000157']  # '000157','000045'

E_list = []
for s in sequences:
    # Read flow images
    F_est = flow_read(os.path.join('..', 'results', 'data_stereo_flow', 'LKflow_' + s + '_10.png'), s, True)
    print s 
    F_gt = flow_read(
        os.path.join('..', '..', 'Datasets', 'data_stereo_flow', 'training', 'flow_noc', s + '_10.png'), s, True)

    # Compute the error map (error at each point)
    # F_val contains the booleans marking the non-occluded pixels
    (E, F_val) = flow_error_map(F_gt, F_est)
    E_list = np.append(E_list, E[F_val != 0])

print('MSE: ' + str(np.mean(E_list)))
print('PEPN: ' + str(np.sum(E_list > 3) * 100. / len(E_list)))
