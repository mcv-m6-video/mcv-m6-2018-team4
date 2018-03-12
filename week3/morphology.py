import cv2
#KERNEL EXAMPLE kernel = np.ones((5,5),np.uint8)

def Erosion(images, kernel, prints,iterations = 1):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing erosion... ')

	for image in range(len(images)):
		results.append(cv2.erode(image,kernel,iterations = iterations))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return results

def Dilation(images, kernel, prints,iterations = 1):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing dilation... ')

	for image in range(len(images)):
		results.append(cv2.dilate(image,kernel,iterations = iterations))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return results

def Opening(images, kernel, prints):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing opening... ')

	for image in range(len(images)):
		results.append(cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return results

def Closing(images, kernel, prints):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing closing... ')

	for image in range(len(images)):
		results.append(cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return results


def Gradient(images, kernel, prints):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing gradient... ')

	for image in range(len(images)):
		results.append(cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return results

def TopHat(images, kernel, prints):
	
	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing top hat... ')

	for image in range(len(images)):
		results.append(cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return results

def BlackHat(images, kernel, prints):

	results = []
	if prints:
		t = time.time()
		sys.stdout.write('Computing black hat... ')

	for image in range(len(images)):
		results.append(cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel))

	if prints:
		elapsed = time.time() - t
		sys.stdout.write(str(elapsed) + ' sec \n')

	return results
