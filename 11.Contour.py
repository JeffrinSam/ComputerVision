#https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
import cv2
import numpy as np
from matplotlib import pyplot as plt

path='E:\Computer Vision stuff\OpenCV\images\images\LP.jpg'
image=cv2.imread(path)


def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

imshow('Input Image', image)
def findcontour(image):
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imshow('After thresholding', th2)

    # Finding Contours
    # Use a copy of your image e.g. edged.copy(), since findContours alters the image
    contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Draw all contours, note this overwrites the input image (inplace operation)
    # Use '-1' as the 3rd parameter to draw all
    cv2.drawContours(image, contours, -1, (0,0,255), thickness = 3)
    imshow("Number of Contours found = " + str(len(contours)), image)

def findContours_Edge(image):
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny Edges
    edged = cv2.Canny(gray, 50, 200)
    imshow('Canny Edges', edged)

    # Finding Contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Draw all contours, note this overwrites the input image (inplace operation)
    # Use '-1' as the 3rd parameter to draw all
    cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
    imshow("Number of Contours found = " + str(len(contours)), image)
    for c in contours:
        print(len(c))

    



findContours_Edge(image)



#findcontour(image)