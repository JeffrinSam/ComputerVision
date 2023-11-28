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

path='E:\Computer Vision stuff\OpenCV\images\images\liberty.jpeg'
image=cv2.imread(path,0)
# Adding comma zero in cv2.imread loads our image in as a grayscaled image
#imshow("Grayscaled Image",image)
print(image)

# Create a matrix of ones, then multiply it by a scaler of 100 
# This gives a matrix with same dimesions of our image with all values being 100
M = np.ones(image.shape, dtype = "uint8") * 100 
imshow("nu",M)

def add_to_matrix(Image,M):
    # We use this to add this matrix M, to our image
    # Notice the increase in brightness
    added = cv2.add(image, M)
    imshow("Increasing Brightness", added)

    # Now if we just added it, look what happens
    added2 = image + M 
    imshow("Simple Numpy Adding Results in Clipping", added2)
    print(added2)

def sub_tomatrix(Image,M):
    # Likewise we can also subtract
    # Notice the decrease in brightness
    subtracted = cv2.subtract(image, M)
    imshow("Subtracted", subtracted)
    subtracted = image - M 
    imshow("Subtracted 2", subtracted)

def shapes_to_image():
    # If you're wondering why only two dimensions, well this is a grayscale image, 
    # Making a square
    square = np.zeros((300, 300), np.uint8)
    cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
    imshow("square", square)

    # Making a ellipse
    ellipse = np.zeros((300, 300), np.uint8)
    cv2.ellipse(ellipse, (150, 150), (150, 150), 10, 0, 180, 255, -1)
    imshow("ellipse", ellipse)
    #cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness)
    return ellipse, square
def bitwise_images(ellipse, square):
    # Shows only where they intersect
    And = cv2.bitwise_and(square, ellipse)
    imshow("AND", And)

    # Shows where either square or ellipse is 
    bitwiseOr = cv2.bitwise_or(square, ellipse)
    imshow("bitwiseOr", bitwiseOr)

    # Shows where either exist by itself
    bitwiseXor = cv2.bitwise_xor(square, ellipse)
    imshow("bitwiseXor", bitwiseXor)

    # Shows everything that isn't part of the square
    bitwiseNot_sq = cv2.bitwise_not(square)
    imshow("bitwiseNot_sq", bitwiseNot_sq)

# Notice the last operation inverts the image totally
bitwise_images(shapes_to_image()[0],shapes_to_image()[1])

#shapes_to_image()
#sub_tomatrix(image,M)