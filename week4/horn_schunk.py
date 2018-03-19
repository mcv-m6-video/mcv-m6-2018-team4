import numpy as np
from skimage import filters
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
 
def optical_flow_hs(t0, t1, alpha):
    h, w = t0.shape[:2]
    gradients = np.gradient(t0)
    dx, dy = gradients[1], gradients[0]
    dt = t1 - t0
    inv_alpha = 1.0 / alpha
    # construct A and b
    b = np.zeros((2, h, w))
    b[0, :, :] = (dt * dx) * inv_alpha
    b[1, :, :] = (dt * dy) * inv_alpha
    b = b.reshape(-1)
    A = np.zeros((h*w*2, h*w*2))
    for row in range(h*w):
        x = row % w
        y = int(row / w)
        # data terms
        A[row, row] = -((dx[y, x] * dx[y, x]) * inv_alpha)
        A[row, row + (w * h)] = -((dy[y, x] * dx[y, x]) * inv_alpha)
        A[row + (w * h), row] = -((dy[y, x] * dx[y, x]) * inv_alpha)
        A[row + (w * h), row + (w * h)] = -((dy[y, x] * dy[y, x]) * inv_alpha)
        # smoothness terms
        A[row, row] -= 4.0
        A[row + (w * h), row + (w * h)] -= 4.0
        for (cy, cx) in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]:
            if cy >= 0 and cy < h and cx >= 0 and cx < w:
                col = (cy * w) + cx
                A[row, col] = 1
                A[row + (w * h), col + (w * h)] = 1
    # solve Ax = b
    A = csc_matrix(A)
    x = spsolve(A, b)
    x = x.reshape((2, h, w))
    u = x[0, :, :]
    v = x[1, :, :]
    return u, v