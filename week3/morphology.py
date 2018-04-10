#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from week3 import *
#KERNEL EXAMPLE kernel = np.ones((5,5),np.uint8)
# Rectangular Kernel
#cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#array([[1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1]], dtype=uint8)
#
## Elliptical Kernel
#cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#array([[0, 0, 1, 0, 0],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [0, 0, 1, 0, 0]], dtype=uint8)
#
## Cross-shaped Kernel
#cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
#array([[0, 0, 1, 0, 0],
#       [0, 0, 1, 0, 0],
#       [1, 1, 1, 1, 1],
#       [0, 0, 1, 0, 0],
#       [0, 0, 1, 0, 0]], dtype=uint8)

#Test 1
#HF,Area,Closing,HF
#Traffic (En la ventana pequeña)
#Fall auc=0.8602 -> 2x2 CUBE
#Highway

#Test 2
#Opening, Closing,HF,Area
#Traffic auc=0.704 -> 17x17 CUBE
#Fall  auc=0.921 -> 3x3 DISK
#Highway auc=0.5976 -> 21x21 DISK

#Test 3
#HF,Area,CLosing
#Traffic
#Fall
#Highway

def precision_recall_curve(train, test, test_GT, ro, conn, p,kernel, prints=True):
	tt = time.time()
	sys.stdout.write('Computing Precision-Recall curve... ')
	#Fall cambiar el 16.2 por 12, el 16.2 solo para highway
	alpha_range = np.around(np.arange(0, 12.2, 1), decimals=2)

	metrics_array = []

	for alpha in alpha_range:
		if prints:
			t = time.time()
			sys.stdout.write("(alpha=" + str(np.around(alpha, decimals=2)) + ") ")
		results = background_substraction(train, test, alpha, ro, False)
		results = hole_filling(results,4, False)
		results = area_filtering(results,4, p, False)
		#results = Opening(results,kernel,False)
		results = Closing(results,kernel,False)
		results = hole_filling(results,4, False)
		metrics = results_evaluation(results, test_GT, False)
		metrics_array.append(metrics)

		if prints:
			elapsed = time.time() - t
			sys.stdout.write(str(elapsed) + ' sec \n')


	precision = np.array(metrics_array)[:, 0]
	recall = np.array(metrics_array)[:, 1]
	auc_val = auc(recall, precision)

	sys.stdout.write("(auc=" + str(np.around(auc_val, decimals=4)) + ") ")

	elapsed = time.time() - tt
	sys.stdout.write(str(elapsed) + ' sec \n')


	if prints:
		plt.plot(recall, precision, color='g')
		print "AUC: "+ str(auc_val)
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.0])
		plt.xlim([0.0, 1.0])
		plt.title("Precision-Recall curve (AUC=" + str(auc_val) + ")" )
		# plt.title("Precision-Recall curve - Fall" )
		plt.show()

	return auc_val


def Erosion(images, kernel, prints,iterations = 1):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing erosion... ')

	for image in images:
		results.append(cv2.erode(image,kernel,iterations = iterations))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return np.array(results)

def Dilation(images, kernel, prints,iterations = 1):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing dilation... ')

	for image in images:
		results.append(cv2.dilate(image,kernel,iterations = iterations))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return np.array(results)

def Opening(images, kernel, prints):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing opening... ')

	for image in images:
		results.append(cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return np.array(results)

def Closing(images, kernel, prints):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing closing... ')

	for image in images:
		results.append(cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return np.array(results)


def Gradient(images, kernel, prints):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing gradient... ')

	for image in images:
		results.append(cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return np.array(results)

def TopHat(images, kernel, prints):
	
	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing top hat... ')

	for image in images:
		results.append(cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return np.array(results)

def BlackHat(images, kernel, prints):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing black hat... ')

	for image in images:
		results.append(cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return np.array(results)

def testing(images,test_GT,kernel,file,Name,i):
	#results = Erosion(images,kernel,True)
	#file.write("Erosion -> Kernel " + Name + ": " + str(i) + "," + str(i)+ "\n")
	#file.write("Recall: " + str(metrics[0] * 100) + "\n")
	#file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	#file.write("F1: " + str(metrics[2] * 100)+ "\n")

	#results = Dilation(images,kernel,True)
	#metrics = results_evaluation(results, test_GT, True)
	#file.write("Dilation -> Kernel " + Name + ": " + str(i) + "," + str(i)+ "\n")
	#file.write("Recall: " + str(metrics[0] * 100) + "\n")
	#file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	#file.write("F1: " + str(metrics[2] * 100)+ "\n")

	#results = Opening(images,kernel,True)
	#metrics = results_evaluation(results, test_GT, True)
	#file.write("Opening -> Kernel " + Name + ": " + str(i) + "," + str(i)+ "\n")
	#file.write("Recall: " + str(metrics[0] * 100) + "\n")
	#file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	#file.write("F1: " + str(metrics[2] * 100)+ "\n")

	#results = Closing(images,kernel,True)
	#metrics = results_evaluation(results, test_GT, True)
	#file.write("Closing -> Kernel " + Name + ": " + str(i) + "," + str(i)+ "\n")
	#file.write("Recall: " + str(metrics[0] * 100) + "\n")
	#file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	#file.write("F1: " + str(metrics[2] * 100)+ "\n")

	#results = Gradient(images,kernel,True)
	#metrics = results_evaluation(results, test_GT, True)
	#file.write("Gradient -> Kernel " + Name + ": " + str(i) + "," + str(i)+ "\n")
	#file.write("Recall: " + str(metrics[0] * 100) + "\n")
	#file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	#file.write("F1: " + str(metrics[2] * 100)+ "\n")
	#results = TopHat(images,kernel,True)
	#metrics = results_evaluation(results, test_GT, True)
	#file.write("TopHat -> Kernel: " + str(i) + "," + str(i)+ "\n")
	#file.write("Recall: " + str(metrics[0] * 100) + "\n")
	#file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	#file.write("F1: " + str(metrics[2] * 100)+ "\n")

	#results = BlackHat(images,kernel,True)
	#metrics = results_evaluation(results, test_GT, True)
	#file.write("BlackHat -> Kernel: " + str(i) + "," + str(i)+ "\n")
	#file.write("Recall: " + str(metrics[0] * 100) + "\n")
	#file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	#file.write("F1: " + str(metrics[2] * 100)+ "\n")

	#results = Closing(images,kernel,True)
	#results = hole_filling(np.array(results),4, True)
	#metrics = results_evaluation(results, test_GT, True)
	#file.write("Closing & HF -> Kernel " + Name + ": " + str(i) + "," + str(i)+ "\n")
	#file.write("Recall: " + str(metrics[0] * 100) + "\n")
	#file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	#file.write("F1: " + str(metrics[2] * 100)+ "\n")
	
	#results = Opening(images,kernel,True)
	#results = Closing(results,kernel,True)
	#results = hole_filling(results,4, True)
	#results = area_filtering(results,4, 100, True)
	#metrics = results_evaluation(results, test_GT, True)
	#file.write("Closing & HF -> Kernel " + Name + ": " + str(i) + "," + str(i)+ "\n")
	#file.write("Recall: " + str(metrics[0] * 100) + "\n")
	#file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	#file.write("F1: " + str(metrics[2] * 100)+ "\n")

	results = Opening(images,kernel,True)
	results = Closing(results,kernel,True)
	results = hole_filling(results,4, True)
	results = area_filtering(results,4, 100, True)
	metrics = results_evaluation(results, test_GT, True)
	file.write("Closing & HF -> Kernel " + Name + ": " + str(i) + "," + str(i)+ "\n")
	file.write("Recall: " + str(metrics[0] * 100) + "\n")
	file.write("Precision: " + str(metrics[1] * 100)+ "\n")
	file.write("F1: " + str(metrics[2] * 100)+ "\n")

def main():

	dataset_name = 'traffic'

	if dataset_name == 'highway':
		frames_range = (1051, 1350)
		alpha = 2.4
		ro = 0.15
		p = 220

	elif dataset_name == 'fall':
		frames_range = (1461, 1560)
		alpha = 3.2
		ro = 0.05
		p = 1800

	elif dataset_name == 'traffic':
		frames_range = ( 951, 1050)
		alpha = 3.5
		ro = 0.15
		p = 330

	else:
		print "Invalid dataset name"
		return

	# Read dataset
	dataset = Dataset(dataset_name,frames_range[0], frames_range[1])

	imgs = dataset.readInput()
	imgs_GT = dataset.readGT()

	# Split dataset
	train = imgs[:len(imgs)/2]
	test = imgs[len(imgs)/2:]
	test_GT = imgs_GT[len(imgs)/2:]
	images = background_substraction(train, test, alpha, ro, True)
	#images = hole_filling(images,4, True)
	#images = area_filtering(images,4, 100, True)
	file = open("logFile_2.txt","w") 
	for i in range(30,50,1):
		print "Kernel: " + str(i) + "," + str(i)
		kernel = np.ones((i,i),np.uint8)
		#testing(images,test_GT,kernel,file,"NORMAL",i)
		precision_recall_curve(train,test,test_GT,ro,4,p,kernel,False)
		#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(i,i))
		#testing(images,test_GT,kernel,file,"MORPH_RECT",i)
		#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i))
		#testing(images,test_GT,kernel,file,"MORPH_ELLIPSE",i)
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(i,i))
		precision_recall_curve(train,test,test_GT,ro,4,p,kernel,False)
		#testing(images,test_GT,kernel,file,"MORPH_CROSS",i)
	file.close()

if __name__ == "__main__":
    main()		
