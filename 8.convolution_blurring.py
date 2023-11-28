import cv2
import numpy as np
from matplotlib import pyplot as plt

def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

path='E:\Computer Vision stuff\OpenCV\images\images\liberty.jpeg'
image=cv2.imread(path)
# Adding comma zero in cv2.imread loads our image in as a grayscaled image
#imshow("Grayscaled Image",image)

def blurring_cov(image):
    # Creating our 3 x 3 kernel
    kernel_3x3 = np.ones((3, 3), np.float32) / 9

    # We use the cv2.fitler2D to conovlve the kernal with an image 
    blurred = cv2.filter2D(image, -1, kernel_3x3)
    imshow('3x3 Kernel Blurring', blurred)
    #filtered_image = cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])

    # Creating our 7 x 7 kernel
    kernel_7x7 = np.ones((10, 10), np.float32) / 100

    blurred2 = cv2.filter2D(image, -1, kernel_7x7)
    imshow('10x10 Kernel Blurring', blurred2)
    
def bluring_gaussian_median(image):
    # Averaging done by convolving the image with a normalized box filter. 
    # This takes the pixels under the box and replaces the central element
    # Box size needs to odd and positive 
    blur = cv2.blur(image, (11,11))
    imshow('Averaging', blur)
    #blurred_image = cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
    # Instead of box filter, gaussian kernel
    Gaussian = cv2.GaussianBlur(image, (5,5), 0)
    imshow('Gaussian Blurring', Gaussian)

    # Takes median of all the pixels under kernel area and central 
    # element is replaced with this median value
    median = cv2.medianBlur(image, 5)
    imshow('Median Blurring', median)

def Bilateral_Filter(image):
    # Bilateral is very effective in noise removal while keeping edges sharp
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
    imshow('fastNlMeansDenoisingColored', dst)
    # Create our shapening kernel, remember it must sum to one 
    kernel_sharpening = np.array([[-1,-1,-1],  [-1, 9,-1], [-1,-1,-1]])
    # applying the sharpening kernel to the image
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    imshow('Sharpened Image', sharpened)


Bilateral_Filter(image)
#bluring_gaussian_median(image)
# blurring_cov(image)