#https://www.researchgate.net/publication/283026260_Background_subtraction_based_on_Gaussian_mixture_models_using_color_and_depth_information
#https://www.researchgate.net/publication/4090386_Improved_Adaptive_Gaussian_Mixture_Model_for_Background_Subtraction
#https://docs.opencv.org/master/de/de1/group__video__motion.html#gac9be925771f805b6fdb614ec2292006d

import cv2
import numpy as np
from matplotlib import pyplot as plt

path="E:\Computer Vision stuff\OpenCV\images\images\walking_short_clip.mp4"
image=cv2.imread(path)
# Define our imshow function 
def imshow(title = "Image", image = None, size = 7):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def bgfgmog(path):
    cap = cv2.VideoCapture(path)

    # Get the height and width of the frame (required to be an interger)
    w = int(cap.get(3)) 
    h = int(cap.get(4))

    # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
    out = cv2.VideoWriter('walking_output_GM.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

    # Initlaize background subtractor
    foreground_background = cv2.bgsegm.createBackgroundSubtractorMOG()

    # Loop once video is successfully loaded
    while True:
    
        ret, frame = cap.read()

        if ret: 
            # Apply background subtractor to get our foreground mask
            foreground_mask = foreground_background.apply(frame)
            out.write(foreground_mask)
            imshow("Foreground Mask", foreground_mask)
        else:
            break

    cap.release()
    out.release()


def bgfggsoc(path):
    cap = cv2.VideoCapture(path)

    # Get the height and width of the frame (required to be an interger)
    w = int(cap.get(3)) 
    h = int(cap.get(4))

    # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
    out = cv2.VideoWriter('walking_output_GM.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

    # Initlaize background subtractor
    foreground_background = cv2.bgsegm.createBackgroundSubtractorGSOC()

    # Loop once video is successfully loaded
    while True:
    
        ret, frame = cap.read()

        if ret: 
            # Apply background subtractor to get our foreground mask
            foreground_mask = foreground_background.apply(frame)
            out.write(foreground_mask)
            imshow("Foreground Mask", foreground_mask)
        else:
            break

    cap.release()
    out.release()

def get_background(path):
    cap = cv2.VideoCapture(path)

    # Get the height and width of the frame (required to be an interger)
    w = int(cap.get(3)) 
    h = int(cap.get(4))

    # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
    out = cv2.VideoWriter('walking_output_GM.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))
    ret, frame = cap.read()

   # Create a flaot numpy array with frame values
    average = np.float32(frame)
    while True:
    
        ret, frame = cap.read()

        if ret: 
            # 0.01 is the weight of image, play around to see how it changes
            cv2.accumulateWeighted(frame, average, 0.01)
      
            # Scales, calculates absolute values, and converts the result to 8-bit
            background = cv2.convertScaleAbs(average)

            imshow('Input', frame)
            imshow('Disapearing Background', background)
            out.write(background)
        else:
             break

    cap.release()
    out.release()   

def KKNfg(path):
    cap = cv2.VideoCapture(path)

    # Get the height and width of the frame (required to be an interfer)
    w = int(cap.get(3))
    h = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorKNN()

    while(1):
        ret, frame = cap.read()

        if ret:
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            imshow('frame', fgmask)
        else:
             break

    cap.release()
    out.release()