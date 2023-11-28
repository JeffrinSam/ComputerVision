import cv2
import numpy as np
from matplotlib import pyplot as plt

path="E:/Computer Vision stuff/OpenCV/images/images/bunchofshapes.jpg"
image=cv2.imread(path)


def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

#imshow('Original Image', image)
def contour(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    print(gray)
    a,b=gray.shape
    imshow('',gray)

    # Find Canny edges
    edged = cv2.Canny(gray, 50, 200)
    imshow('Canny Edges', edged)

    # Find contours and print how many were found
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Draw all contours over blank image
    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    imshow('All Contours-Number of contours found =' +str(len(contours)), image)
    return contours


def momentsof(image,c,A):
    # Let's print the areas of the contours before sorting
    print("Contor Areas before sorting...")
    print(get_contour_areas(c))

    # Sort contours large to small by area
    sorted_contours = sorted(c, key=cv2.contourArea, reverse=True)

    print("Contor Areas after sorting...") 
    print(get_contour_areas(sorted_contours))

    # Iterate over our contours and draw one at a time
    for (i,c) in enumerate(sorted_contours):
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.drawContours(image, [c], -1, (255,0,0), 3)
    imshow('Contours by area', image)

def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

# Functions we'll use for sorting by position
def x_cord_contour(c):
    """Returns the X cordinate for the contour centroid"""
    if cv2.contourArea(c) > 10:
        M = cv2.moments(c)
        return (int(M['m10']/M['m00']))
    else:
        pass
    
def label_contour_center(image, c):
    """Places a red circle on the centers of contours"""
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # Draw the countour number on the image
    cv2.circle(image,(cx,cy), 10, (0,0,255), -1)
    return image


#c=contour(image)
#A=get_contour_areas(c)
#momentsof(image,c,A)
#print('Cont'+str(A))


def contour2():
    path="E:/Computer Vision stuff/OpenCV/images/images/house.jpg"
    image=cv2.imread(path)    
    orig_image = image.copy()
    imshow('Original Image', orig_image)
    
    # Grayscale and binarize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours 
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    copy = image.copy()

    # Iterate through each contour 
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.drawContours(image, [c], 0, (0, 255, 0), 2)

    imshow('Drawing of Contours', image)
    imshow('Bounding Rectangles', orig_image)
    # Iterate through each contour and compute the approx contour
    for c in contours:
        # Calculate accuracy as a percent of the contour perimeter
        accuracy = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, accuracy, True)
        cv2.drawContours(copy, [approx], 0, (0, 255, 0), 2)

    imshow('Approx Poly DP', copy)

#contour2()

def covexContour():
    path="E:/Computer Vision stuff/OpenCV/images/images/hand.jpg"
    image=cv2.imread(path)   
    orginal_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imshow('Original Image', image)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 176, 255, 0)

    # Find contours 
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, 0, (0, 255, 0), 2)
    imshow('Contours of Hand', image)


    # Sort Contors by area and then remove the largest frame contour
    n = len(contours) - 1
    contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]
    print(n, contours)
    # Iterate through contours and draw the convex hull
    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(orginal_image, [hull], 0, (0, 255, 0), 2)
        
    imshow('Convex Hull', orginal_image)

#covexContour()
def matchingcontour():
    # Load the shape template or reference image
    template = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/4star.jpg',0)
    imshow('Template', template)

    # Load the target image with the shapes we're trying to match
    target = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/shapestomatch.jpg')
    target_gray = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)

    # Threshold both images first before using cv2.findContours
    ret, thresh1 = cv2.threshold(template, 127, 255, 0)
    ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

    # Find contours in template
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # We need to sort the contours by area so that we can remove the largest
    # contour which is the image outline
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # We extract the second largest contour which will be our template contour
    template_contour = contours[1]

    # Extract contours from second target image
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # Iterate through each contour in the target image and 
        # use cv2.matchShapes to compare contour shapes
        match = cv2.matchShapes(template_contour, c, 3, 0.0)
        print(match)
        # If the match value is less than 0.15 we
        if match < 0.15:
            closest_contour = c
        else:
            closest_contour = [] 
                    
    cv2.drawContours(target, [closest_contour], -1, (0,255,0), 3)
    imshow('Output', target)

matchingcontour()