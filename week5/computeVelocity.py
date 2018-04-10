
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure

from getXY import getXY

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

    for detection in detections:
        for i in range(1,len(detection.posList)):
            p1 = detection.posList[i-1]
            p2 = detection.posList[i]

            p1_h = np.array([p1[0], p1[1], 1])
            p2_h = np.array([p2[0], p2[1], 1])

            p1_aereal = np.matmul(H, p1_h.transpose())
            p1_aereal = p1_aereal / p1_aereal[2]

            p2_aereal = np.matmul(H, p2_h.transpose())
            p2_aereal = p2_aereal / p2_aereal[2]

            distance = (p2_aereal - p1_aereal)
            distance = np.sqrt((distance[0]**2)+(distance[1]**2))*scale_factor

            time = (detection.framesList[i] - detection.framesList[i-1])*(1./25)

            detection.updateVelocity(distance/time)

            # print (distance/time)*3.6

    return 0



# %% 5.4 Distancia a furgoneta %%%%%%%%%%%%%%%%%%%%%%%
# clear all, close all
# disp('1. Carrega la imatge');
# [filename, pathname] = uigetfile( ...
#        {'*.jpg;*.tif;*.bmp', 'Image Files (*.jpg, *.bmp, *.tif)';
#         '*.*',  'All Files (*.*)'}, ...
#         'Selecciona una imatge');
# ima = imread(strcat(pathname,filename));
# figure(1)
# imshow(ima,[]);
#
# % punt de fuga i homografia %
# [X Y] = ginput(4);
# figure(1), hold on
# plot(X(1:2),Y(1:2),'r:')
# plot(X(3:4),Y(3:4),'r:')
# plot(X,Y,'g+')
#
# p1 = [X(1) Y(1) 1];
# p2 = [X(2) Y(2) 1];
# p3 = [X(3) Y(3) 1];
# p4 = [X(4) Y(4) 1];
#
# r1=cross(p1,p2);
# r2=cross(p3,p4);
# v=cross(r1,r2);
# v=v/v(3);
#
# plot(v(1),v(2),'rx')
#
# H= [1 -v(1)/v(2) 0;
#     0     1      0;
#     0  -1/v(2)   1];
#
# % entrada de valors de calibracio %
# distanciaBase = input('Distancia del cotxe a lultima fila de la imatge (m): ');
# text(size(ima,2)/2,size(ima,1),sprintf('%4.2f m',distanciaBase));
#
# disp('Marca dos punts que defineixin un segment de longitud coneguda sobre el pla');
# [X Y] = ginput(2);
# plot(X,Y,'g+:')
#
# longitudSegment = input('Introdueix el valor de la longitud del segment per a calibrar (m): ');
# text(mean(X),mean(Y),sprintf('%4.2f m',longitudSegment));
#
# extremsSegment = [X(1) X(2);
#                   Y(1) Y(2);
#                     1    1];   % coord. homogenies
# extremsSegmentAfi = H*extremsSegment;
# extremsSegmentAfi(:,1) = extremsSegmentAfi(:,1)/extremsSegmentAfi(3,1);
# extremsSegmentAfi(:,2) = extremsSegmentAfi(:,2)/extremsSegmentAfi(3,2);
#
# FactorEscala = longitudSegment/abs(extremsSegmentAfi(2,2)-extremsSegmentAfi(2,1));
#
# % Calcul de distancies %
#
# puntBase = [size(ima,2)/2,size(ima,1),1]; % coord. homogenies
# puntBaseAfi = H*puntBase';
# puntBaseAfi = puntBaseAfi/puntBaseAfi(3);
#
# disp('Selecciona diferents punts. Pulsa Enter per acabar');
# [X Y] = ginput;
# plot(X,Y,'go')
#
# n = size(X,1);
#
# punts =  [ X'; Y'; ones(1,n)];% homogeneous coordinates
# puntsAfi = H*punts;
#
# for i=1:n
#     puntsAfi(:,i) = puntsAfi(:,i)/puntsAfi(3,i);
# end
#
# distanciesPunts= FactorEscala*abs(puntsAfi(2,:)-puntBaseAfi(2))+distanciaBase;
#
# textDistancia = num2str(distanciesPunts','%4.2f') ;
# text(punts(1,:)+5,punts(2,:),textDistancia)
# hold off