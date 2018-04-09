import cv2
import numpy as np

a = np.array([0, 0], dtype='float32')


def getXY(image):
    global a

    # Set mouse CallBack event
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', getxy)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    b = a[1:,:].copy()

    return b


# define the event
def getxy(event, x, y, flags, param):
    global a

    if event == cv2.EVENT_LBUTTONDOWN:
        a = np.vstack([a, np.hstack([x, y])])
        print "(row, col) = ", (x, y)
    return