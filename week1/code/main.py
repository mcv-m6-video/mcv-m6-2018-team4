import cv2
import functions as fc
import os



def main():
    gt_imgs = fc.readImages("../highway/groundtruth","png")
    A_imgs = fc.readImages("../results/highway/A","png")
    B_imgs = fc.readImages("../results/highway/B","png")

    [TP, TN, FP, FN] = fc.getMetrics(A_imgs,gt_imgs)
    print "Computing Metrics with TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)


#A_CMatrix  = fc.ConfusionMatrix(gt_imgs,A_imgs)
	#A_Metrics  = fc.Metrics(A_CMatrix[0],A_CMatrix[1],A_CMatrix[2],A_CMatrix[3])
	#B_CMatrix  = fc.ConfusionMatrix(gt_imgs,B_imgs)
	#B_Metrics  = fc.Metrics(B_CMatrix[0],B_CMatrix[1],B_CMatrix[2],B_CMatrix[3])

if __name__ == "__main__":
    main()