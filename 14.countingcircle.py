#http://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html
import cv2
import numpy as np
from matplotlib import pyplot as plt

path="E:/Computer Vision stuff/OpenCV/images/images/blobs.jpg"
image=cv2.imread(path)


def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
imshow('',image)

def detectcircleandcount():
    # Load image
    image = cv2.imread("E:/Computer Vision stuff/OpenCV/images/images/blobs.jpg", 0)
    imshow('Original Image',image)

    # Intialize the detector using the default parameters
    detector = cv2.SimpleBlobDetector_create()
    
    # Detect blobs
    keypoints = detector.detect(image)
    
    # Draw blobs on our image as red circles
    blank = np.zeros((1,1)) 
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Total Number of Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

    # Display image with blob keypoints
    imshow("Blobs using default parameters", blobs)

    # Set our filtering parameters
    # Initialize parameter settiing using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 100

    # Set Circularity filtering parameters
    params.filterByCircularity = True 
    params.maxCircularity = 0.9

    # Set Convexity filtering parameters
    params.filterByConvexity = False
    params.minConvexity = 0.2
        
    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
        
    # Detect blobs
    keypoints = detector.detect(image)

    # Draw blobs on our image as red circles
    blank = np.zeros((1,1)) 
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # Show blobs
    imshow("Filtering Circular Blobs Only", blobs)

#detectcircleandcount()


def waldodet() :
    template = cv2.imread("E:/Computer Vision stuff/OpenCV/images/images/waldo.jpg")
    imshow('Original Image',template)
    # Load input image and convert to grayscale
    image = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/WaldoBeach.jpg')
    imshow('Where is Waldo?', image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Template image
    template = cv2.imread('./images/waldo.jpg',0)

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    #Create Bounding Box
    top_left = max_loc
    bottom_right = (top_left[0] + 50, top_left[1] + 50)
    cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

    imshow('Where is Waldo?', image)
waldodet()