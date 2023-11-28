
#Dilation – Adds pixels to the boundaries of objects in an image
#Erosion – Removes pixels at the boundaries of objects in an image
#Opening - Erosion followed by dilation
#Closing - Dilation followed by erosion
import cv2
import numpy as np
from matplotlib import pyplot as plt

path='E:\Computer Vision stuff\OpenCV\images\images\opencv_inv.png'
image=cv2.imread(path,0)

def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def dilation(image):
    # Let's define our kernel size
    kernel = np.ones((5,5), np.uint8)

    # Now we erode
    erosion = cv2.erode(image, kernel, iterations = 1)
    imshow('Erosion', erosion)

    # Dilate here
    dilation = cv2.dilate(image, kernel, iterations = 2)
    imshow('Dilation', dilation)

    # Opening - Good for removing noise
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    imshow('Opening',opening)

    # Closing - Good for removing noise
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    imshow('Closing',closing)

def Edgedetection():
    image = cv2.imread('E:\Computer Vision stuff\OpenCV\images\images\londonxmas.jpeg',0)
    # Canny Edge Detection uses gradient values as thresholds
    # The first threshold gradient
    canny = cv2.Canny(image, 50, 120)
    imshow('Canny 1', canny)

    # Wide edge thresholds expect lots of edges
    canny = cv2.Canny(image, 10, 200)
    imshow('Canny Wide', canny)

    # Narrow threshold, expect less edges 
    canny = cv2.Canny(image, 200, 240)
    imshow('Canny Narrow', canny)

    canny = cv2.Canny(image, 60, 110)
    imshow('Canny 4', canny)

    ##  Then, we need to provide two values: threshold1 and threshold2. 
    # Any gradient value larger than threshold2
    # is considered to be an edge. Any value below threshold1 is considered not to be an edge. 
    #Values in between threshold1 and threshold2 are either classiﬁed as edges or non-edges based on how their 
    #intensities are “connected”. In this case, any gradient values below 60 are considered non-edges
    #whereas any values above 120 are considered edges.

def autoCanny():
  # Finds optimal thresholds based on median image pixel intensity
  image = cv2.imread('E:\Computer Vision stuff\OpenCV\images\images\londonxmas.jpeg',0)
  blurred_img = cv2.blur(image, ksize=(5,5))
  med_val = np.median(image) 
  lower = int(max(0, 0.66 * med_val))
  upper = int(min(255, 1.33 * med_val))
  edges = cv2.Canny(image=image, threshold1=lower, threshold2=upper)
  return edges

auto_canny = autoCanny()
imshow("auto canny", auto_canny)

#Edgedetection()
#imshow('',image)
#dilation(image)