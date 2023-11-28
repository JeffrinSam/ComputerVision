import cv2
import numpy as np
from matplotlib import pyplot as plt

path="E:/Computer Vision stuff/OpenCV/images/images/soduku.jpg"
image=cv2.imread(path)


def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
imshow('',image)
def houghlinesfind(image):
    # Grayscale and Canny Edges extracted
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

    # Run HoughLines using a rho accuracy of 1 pixel
    # theta accuracy of np.pi / 180 which is 1 degree
    # Our line threshold is set to 240 (number of points on line)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 240)
    print(lines,lines.shape)
    # We iterate through each line and convert it to the format
    # required by cv2.lines (i.e. requiring end points)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    imshow('Hough Lines', image)

#houghlinesfind(image)

def houghp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

    # Again we use the same rho and theta accuracies
    # However, we specific a minimum vote (pts along line) of 100
    # and Min line length of 3 pixels and max gap between lines of 25 pixels
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 3, 25)
    print(lines.shape)

    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

    imshow('Probabilistic Hough Lines', image)


#houghp(image)

def houghcircle():
    image = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/Circles_Packed_In_Square_11.png')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 25)

    cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100)

    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(image,(i[0], i[1]), i[2], (0, 0, 255), 5)
        
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 255, 255), 8)

    imshow('Detected circles', image)

#houghcircle()

def blobdetection():
    # Read image
    image = cv2.imread("E:/Computer Vision stuff/OpenCV/images/images/Sunflowers.jpg")
    imshow("Original", image)

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()
    
    # Detect blobs.
    keypoints = detector.detect(image)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of
    # the circle corresponds to the size of blob
    blank = np.zeros((1,1)) 
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    # Show keypoints
    imshow("Blobs", blobs)

blobdetection()