import cv2
import numpy as np
from matplotlib import pyplot as plt

path="E:/Computer Vision stuff/OpenCV/images/images/chess.jpg"
image=cv2.imread(path)


def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
imshow('',image)

def face():
    # We point OpenCV's CascadeClassifier function to where our 
    # classifier (XML file format) is stored
    face_classifier = cv2.CascadeClassifier('E:\Computer Vision stuff\OpenCV\haarcascades\Haarcascades\haarcascade_frontalface_default.xml')

    # Load our image then convert it to grayscale
    image = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/Trump.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Our classifier returns the ROI of the detected face as a tuple
    # It stores the top left coordinate and the bottom right coordiantes
    faces = face_classifier.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

    # When no faces detected, face_classifier returns and empty tuple
    # We iterate through our faces array and draw a rectangle
    # over each face in faces
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)

    imshow('Face Detection', image)

face()