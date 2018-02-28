import cv2
import functions as fc
import numpy as np
import os



def main():
	ori_imgs = fc.readImages("../highway/groundtruth","png")
	A_imgs   = fc.readImages("../results/highway/A","png")
	B_imgs   = fc.readImages("../results/highway/B","png")

	A_CMatrix,TPFv,TPGTv = fc.ConfusionMatrix(ori_imgs,A_imgs)
	A_Metrics            = fc.Metrics(A_CMatrix[0],A_CMatrix[1],A_CMatrix[2],A_CMatrix[3])
 	fc.showMultipleGraphic([np.arange(0, 200,1),np.arange(0, 200,1)],[np.array(TPFv),np.array(TPGTv)],"TP & TF vs time",'TP','TF')

	B_CMatrix,TPFv,TPGTv = fc.ConfusionMatrix(ori_imgs,B_imgs)
	B_Metrics,vTP        = fc.Metrics(B_CMatrix[0],B_CMatrix[1],B_CMatrix[2],B_CMatrix[3])
	fc.showMultipleGraphic([np.arange(0, 200,1),np.arange(0, 200,1)],[np.array(TPFv),np.array(TPGTv)],"TP & TF vs time",'TP','TF')


if __name__ == "__main__":
    main()