#https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
#https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html
#https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_local

def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

path='E:\Computer Vision stuff\OpenCV\images\images\scan.jpeg'
image=cv2.imread(path,0)

def threshold_image(image):
    # Values below 127 goes to 0 or black, everything above goes to 255 (white)
    ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    print(ret,thresh1)
    imshow('1 Threshold Binary @ 127', thresh1)

    # Values below 127 go to 255 and values above 127 go to 0 (reverse of above)
    ret,thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    imshow('2 Threshold Binary Inverse @ 127', thresh2)

    # Values above 127 are truncated (held) at 127 (the 255 argument is unused)
    ret,thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    imshow('3 THRESH TRUNC @ 127', thresh3)

    # Values below 127 go to 0, above 127 are unchanged  
    ret,thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    imshow('4 THRESH TOZERO @ 127', thresh4)

    # Reverse of the above, below 127 is unchanged, above 127 goes to 0
    ret,thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
    imshow('5 THRESH TOZERO INV @ 127', thresh5)

def Adaptive_threshold_image(image):
    # Values below 127 goes to 0 (black, everything above goes to 255 (white)
    ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    imshow('Threshold Binary', thresh1)

    # It's good practice to blur images as it removes noise
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Using adaptiveThreshold
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,5) 
    imshow("Adaptive Mean Thresholding", thresh) 

    _, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imshow("Otsu's Thresholding", th2) 

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image, (5,5), 0)
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imshow("Guassian Otsu's Thresholding", th3) 


def skimage_A(image):
    # We get the Value component from the HSV color space 
    # then we apply adaptive thresholdingto 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    V = hsv_image[:, :, 2]
    T = threshold_local(V, 25, offset=5, method="gaussian")
    print(V)
    # Apply the threshold operation 
    thresh = (V > T).astype("uint8") * 255
    imshow("threshold_local", thresh)

skimage_A(image=cv2.imread(path))
#Adaptive_threshold_image(image)
#threshold_image(image)
