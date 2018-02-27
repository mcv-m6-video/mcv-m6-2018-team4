import cv2
import functions as fc
import os



def main():
	ori_imgs   = fc.readImages("../highway/groundtruth","png")
	A_imgs     = fc.readImages("../results/highway/A","png")
	B_imgs     = fc.readImages("../results/highway/B","png")
	A_CMatrix  = fc.ConfusionMatrix(ori_imgs,A_imgs)
	A_Metrics  = fc.Metrics(A_CMatrix[0],A_CMatrix[1],A_CMatrix[2],A_CMatrix[3])
	B_CMatrix  = fc.ConfusionMatrix(ori_imgs,B_imgs)
	B_Metrics  = fc.Metrics(B_CMatrix[0],B_CMatrix[1],B_CMatrix[2],B_CMatrix[3])

if __name__ == "__main__":
    main()