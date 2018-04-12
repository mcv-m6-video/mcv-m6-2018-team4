
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure

from getXY import getXY

def computeVelocity_own(images, detections):

    # Get manually 4 points that form 2 paralel lines
    image0 = images[0]

    points = getXY(image0)

    p1 = np.array([points[0,0], points[0,1], 1])
    p2 = np.array([points[1,0], points[1,1], 1])

    # H =np.array([[-0.00933262047476400,	1.25094484415116,	-204.171484829449    ],
    #              [ 0.322802008471952,  	0.492556909522902,	 -42.8032678476063   ],
    #              [-1.97412767096888e-06,0.00224106725153738,  -0.0266465643937424]])

    H = np.array([[-0.729452734014885, -0.297141608098521, 85.4043267941782],
                  [0.0119609003945710, -1.19656747018213,   131.687711887598],
                  [5.57976613691146e-05, -0.00271531878836268,   0.0424222720888309]])

    im_warped = cv2.warpPerspective(image0, H, (image0.shape[1],image0.shape[0]))

    plt.imshow(im_warped)
    plt.show()

    line_longitude = 2.5 # in meters
    p1_aereal = np.matmul(H,p1.transpose())
    p1_aereal = p1_aereal/p1_aereal[2]

    p2_aereal = np.matmul(H,p2.transpose())
    p2_aereal = p2_aereal/p2_aereal[2]

    scale_factor = line_longitude/abs(p1_aereal[1]-p2_aereal[1]);
    fps = 20

    print "scale factor: " + str(scale_factor)
    frame_step = 1
    for detection in detections:
        if(len(detection.posList)>frame_step):
            for i in range(frame_step,len(detection.posList),frame_step):
                p1 = detection.posList[i-frame_step]
                p2 = detection.posList[i]

                p1_h = np.array([p1[0], p1[1], 1])
                p2_h = np.array([p2[0], p2[1], 1])

                p1_aereal = np.matmul(H, p1_h.transpose())
                p1_aereal = p1_aereal / p1_aereal[2]

                p2_aereal = np.matmul(H, p2_h.transpose())
                p2_aereal = p2_aereal / p2_aereal[2]

                distance = (p2_aereal - p1_aereal)
                # distance = np.sqrt((distance[0]**2)+(distance[1]**2))*scale_factor
                distance = abs(distance[1])*scale_factor

                time = (detection.framesList[i] - detection.framesList[i-frame_step])*(1./fps)

                detection.updateVelocity(distance/time,frame_step)

            # print (distance/time)*3.6

    return 0

def computeVelocity(images, detections):

    # Get manually 4 points that form 2 paralel lines
    image0 = images[0]

    points = getXY(image0)

    p1 = np.array([points[0,0], points[0,1], 1])
    p2 = np.array([points[1,0], points[1,1], 1])
    p3 = np.array([points[2,0], points[2,1], 1])
    p4 = np.array([points[3,0], points[3,1], 1])

    l1 = np.cross(p1,p2)
    l2 = np.cross(p3,p4)

    # Compute the vanishing point
    v = np.cross(l1,l2)
    v = v/v[2]

    # Create homography for aereal view
    H = np.array([[1, -v[0]/v[1], 0],
                 [0,    1,       0],
                 [0,  -1/v[1],   1]])

    im_warped = cv2.warpPerspective(image0, H, (image0.shape[1],image0.shape[0]))

    plt.imshow(im_warped)
    plt.show()

    line_longitude = 2.5 # in meters
    p1_aereal = np.matmul(H,p1.transpose())
    p1_aereal = p1_aereal/p1_aereal[2]

    p2_aereal = np.matmul(H,p2.transpose())
    p2_aereal = p2_aereal/p2_aereal[2]

    scale_factor = line_longitude/abs(p1_aereal[1]-p2_aereal[1]);

    print "scale factor: " + str(scale_factor)
    frame_step = 1
    for detection in detections:
        if(len(detection.posList)>frame_step):
            for i in range(frame_step,len(detection.posList),frame_step):
                p1 = detection.posList[i-frame_step]
                p2 = detection.posList[i]

                p1_h = np.array([p1[0], p1[1], 1])
                p2_h = np.array([p2[0], p2[1], 1])

                p1_aereal = np.matmul(H, p1_h.transpose())
                p1_aereal = p1_aereal / p1_aereal[2]

                p2_aereal = np.matmul(H, p2_h.transpose())
                p2_aereal = p2_aereal / p2_aereal[2]

                distance = (p2_aereal - p1_aereal)
                distance = np.sqrt((distance[0]**2)+(distance[1]**2))*scale_factor

                time = (detection.framesList[i] - detection.framesList[i-frame_step])*(1./30)
                detection.updateVelocity(distance/time,frame_step)

            # print (distance/time)*3.6

    return 0