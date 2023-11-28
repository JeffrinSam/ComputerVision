import cv2
import numpy as np
from matplotlib import pyplot as plt

path="E:/Computer Vision stuff/OpenCV/images/images/Hillary.jpg"
image=cv2.imread(path)
# Define our imshow function 
def imshow(title = "Image", image = None, size = 7):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

image = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/truck.jpg')

# define range of BLUE color in HSV
lower = np.array([90,0,0])
upper = np.array([135,255,255])

# Convert image from RBG/BGR to HSV so we easily filter
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Use inRange to capture only the values between lower & upper 
mask = cv2.inRange(hsv_img, lower, upper)

# Perform Bitwise AND on mask and our original frame
res = cv2.bitwise_and(image, image, mask=mask)

imshow('Original', image)  
imshow('mask', mask)
imshow('Filtered Color Only', res)

def filteringface(path):
    image = cv2.imread(path)

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,0,0])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,0,0])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join masks
    mask = mask0+mask1

    # Perform Bitwise AND on mask and our original frame
    res = cv2.bitwise_and(image, image, mask=mask)

    imshow('Original', image)  
    imshow('mask', mask)
    imshow('Filtered Color Only', res)

filteringface(path)