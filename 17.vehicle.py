import cv2
import numpy as np
from matplotlib import pyplot as plt

path="E:\Computer Vision stuff\OpenCV\haarcascades\Haarcascades\haarcascade_fullbody.xml"

def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def first_image():
    # Create our video capturing object
    cap = cv2.VideoCapture('E:\Computer Vision stuff\OpenCV\images\images\walking.avi')

    # Load our body classifier
    body_classifier = cv2.CascadeClassifier("E:\Computer Vision stuff\OpenCV\haarcascades\Haarcascades\haarcascade_fullbody.xml")

    # Read first frame
    ret, frame = cap.read()

    # Ret is True if successfully read
    if ret: 

        #Grayscale our image for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass frame to our body classifier
        bodies = body_classifier.detectMultiScale(gray, 1.1, 4)

        # Extract bounding boxes for any bodies identified
        for (x,y,w,h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    # Release our video capture
    cap.release()   
    imshow("Pedestrian Detector", frame)

def image_section():
    # Create our video capturing object
    cap = cv2.VideoCapture('E:\Computer Vision stuff\OpenCV\images\images\walking.avi')

    # Get the height and width of the frame (required to be an interfer)
    w = int(cap.get(3))
    h = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'walking_output.avi' file.
    out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

    body_detector = cv2.CascadeClassifier("E:\Computer Vision stuff\OpenCV\haarcascades\Haarcascades\haarcascade_fullbody.xml")

    # Loop once video is successfully loaded
    while(True):

        ret, frame = cap.read()
        if ret: 

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Pass frame to our body classifier
            bodies = body_detector.detectMultiScale(gray, 1.1, 3)

            # Extract bounding boxes for any bodies identified
            for (x,y,w,h) in bodies:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Write the frame into the file 'output.avi'
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

def vechile_detectMultiScale2() :
    # Create our video capturing object
    cap = cv2.VideoCapture('images\images\cars.avi')

    # Get the height and width of the frame (required to be an interfer)
    w = int(cap.get(3))
    h = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('cars_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

    vehicle_detector = cv2.CascadeClassifier('E:\Computer Vision stuff\OpenCV\haarcascades\Haarcascades\haarcascade_car.xml')

    # Loop once video is successfully loaded
    while(True):

        ret, frame = cap.read()
        if ret: 

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Pass frame to our body classifier
            vehicles = vehicle_detector.detectMultiScale(gray, 1.1, 1)

            # Extract bounding boxes for any bodies identified
            for (x,y,w,h) in vehicles:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Write the frame into the file 'output.avi'
            out.write(frame)
        else:
            break

    cap.release()
    out.release()


vechile_detectMultiScale2()