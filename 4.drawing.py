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

def main():
    image=np.zeros((512,512,3),np.uint8)
    image_gray=128*np.ones((512,512),np.uint8)
    image_white=255*np.ones((512,512),np.uint8)
    imshow("black",image)
    imshow("gray",image_gray)
    imshow("white",image_white)
    cv2.line(image,(0,0),(511,511),(255,0,0),3)
    imshow("line",image)
    #syntax:cv2.line(image, start_point, end_point, color, thickness) 
    cv2.rectangle(image,(100,100),(250,250),(255,0,0),3)
    imshow("rec",image)
    #cv2.rectangle(image, start_point, end_point, color, thickness) 

main()




'''cv2.arrowedLine(image, start_point, end_point, color, thickness, line_type, shift, tipLength)
cv2.ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, color , thickness, lineType, shift)
cv2.circle(image, center_coordinates, radius, color, thickness)
cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType, bottomLeftOrigin)
cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)'''