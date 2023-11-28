#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
import cv2
import numpy as np
from matplotlib import pyplot as plt

path="E:\Computer Vision stuff\OpenCV\images\slow.flv"

# Define our imshow function 
def imshow(title = "Image", image = None, size = 7):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def objecttracking():
    # Load video stream, short clip
    cap = cv2.VideoCapture('E:\Computer Vision stuff\OpenCV\images\walking_short_clip.mp4')

    # Load video stream, long clip
    cap = cv2.VideoCapture('E:\Computer Vision stuff\OpenCV\images\walking.avi')

    # Get the height and width of the frame (required to be an interger)
    width = int(cap.get(3)) 
    height = int(cap.get(4))

    # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
    out = cv2.VideoWriter('optical_flow_walking.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

    # Set parameters for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7,blockSize = 7 )

    # Set parameters for lucas kanade optical flow
    lucas_kanade_params = dict( winSize  = (15,15),maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    # Used to create our trails for object movement in the image 
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Find inital corner locations
    prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_frame)

    while(1):
        ret, frame = cap.read()

        if ret == True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray,  frame_gray,  prev_corners,  None,  **lucas_kanade_params)

            # Select and store good points
            good_new = new_corners[status==1]
            good_old = prev_corners[status==1]

            # Draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
          
            img = cv2.add(frame,mask)

            # Save Video
            out.write(img)
            # Show Optical Flow
            imshow('Optical Flow - Lucas-Kanade',img)

             # Now update the previous frame and previous points
            prev_gray = frame_gray.copy()
            prev_corners = good_new.reshape(-1,1,2)

        else:
            break
    
    cap.release()
    out.release()
    
#objecttracking()

# Load video stream, short clip
#cap = cv2.VideoCapture('walking_short_clip.mp4')

# Load video stream, long clip
cap = cv2.VideoCapture('E:\Computer Vision stuff\OpenCV\images\walking.avi')

# Get the height and width of the frame (required to be an interger)
width = int(cap.get(3)) 
height = int(cap.get(4))

# Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
out = cv2.VideoWriter('dense_optical_flow_walking.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

# Get first frame
ret, first_frame = cap.read()
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[...,1] = 255

while True:
    
    # Read of video file
    ret, frame2 = cap.read()

    if ret == True:
      next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

      # Computes the dense optical flow using the Gunnar Farneback’s algorithm
      flow = cv2.calcOpticalFlowFarneback(previous_gray, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

      # use flow to calculate the magnitude (speed) and angle of motion
      # use these values to calculate the color to reflect speed and angle
      magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
      hsv[...,0] = angle * (180 / (np.pi/2))
      hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
      final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

      # Save Video
      out.write(final)
      # Show our demo of Dense Optical Flow
      #imshow('Dense Optical Flow', final)
      
      # Store current image as previous image
      previous_gray = next

    else:
      break
    
cap.release()
out.release()