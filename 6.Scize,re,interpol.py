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

def resize_image(image):
    image_scaled =cv2.resize(image,None,fx=0.75,fy=0.75)
    imshow('0.75',image_scaled)
    image_scaled2 =cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    imshow('CUBIC',image_scaled2)
    image_scaled3 =cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4)
    imshow('ANCZOS4',image_scaled3)
    image_scaled4 =cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    imshow('NEAREST',image_scaled4)
    image_scaled5 =cv2.resize(image,(640,480),interpolation=cv2.INTER_AREA)
    imshow('AREA',image_scaled5)

def resolution_mod(image):
    small=cv2.pyrDown(image)
    imshow('resolutiondown',small)
    large=cv2.pyrUp(image)
    imshow("resup",large)


def main():
    path='E:\Computer Vision stuff\OpenCV\images\images\oxfordlibrary.jpeg'
    image=cv2.imread(path)
    imshow('trail',image)
    h,w,d=image.shape
    start_row,start_col,end_row,end_col=int(h*.25),int(w*.25),int(h*.55),int(w*.55)
    crop=image[start_row:end_row,start_col:end_col]
    copy=image.copy()
    cv2.rectangle(copy,(start_row,start_col),(end_row,end_col),(255,0,0),7)
    imshow("cro",crop)
    imshow("copy",copy)


path='E:\Computer Vision stuff\OpenCV\images\images\oxfordlibrary.jpeg'
image=cv2.imread(path)
resolution_mod(image)
resize_image(image)

'''Re-sizing is a simple function that we execute using the cv2.resize function, it's arguments are:

cv2.resize(image, dsize(output image size), x scale, y scale, interpolation)

if dsize is None the output image is calculated as a function of scaling using x & y scale 

#### **List of Interpolation Methods:**
cv2.INTER_AREA - Good for shrinking or down sampling
cv2.INTER_NEAREST - Fastest
cv2.INTER_LINEAR - Good for zooming or up sampling (default)
cv2.INTER_CUBIC - Better
cv2.INTER_LANCZOS4 - Best'''

