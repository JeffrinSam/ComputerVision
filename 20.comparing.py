import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim


path="E:/Computer Vision stuff/OpenCV/images/images/input.jpg"
image=cv2.imread(path)


def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def mse(image1, image2):
	# Images must be of the same dimension
	error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
	error /= float(image1.shape[0] * image1.shape[1])

	return error

fireworks1 = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/fireworks.jpeg')
fireworks2 = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/fireworks2.jpeg')

M = np.ones(fireworks1.shape, dtype = "uint8") * 100 
fireworks1b = cv2.add(fireworks1, M)

imshow("fireworks 1", fireworks1)
imshow("Increasing Brightness", fireworks1b)
imshow("fireworks 2", fireworks2)

def compare(image1, image2):
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  print('MSE = {:.2f}'.format(mse(image1, image2)))
  print('SSIM = {:.2f}'.format(compare_ssim(image1, image2)))
 
compare(fireworks1, fireworks1)
compare(fireworks1, fireworks2)
compare(fireworks1, fireworks1b)
compare(fireworks2, fireworks1b)