import numpy as np
from skimage import filters
 
def optical_flow_lk(t0, t1, sigma):
    # setup the local linear systems of equations
    gradients = np.gradient(t0)
    dx, dy = gradients[1], gradients[0]
    dt = t1 - t0
    A00 = filters.gaussian(dx * dx, sigma)
    A11 = filters.gaussian(dy * dy, sigma)
    A10 = filters.gaussian(dx * dy, sigma)
    A01 = A10
    b0 = -filters.gaussian(dx * dt, sigma)
    b1 = -filters.gaussian(dy * dt, sigma)
 
    # solve the local linear systems of equations via 2x2 matrix inversion
    determinant = 1.0 / ((A00 * A11) - (A10 * A10))
    u = determinant * ((b0 * A11)  + (b1 * -A01))
    v = determinant * ((b0 * -A10) + (b1 * A00))
    return u, v