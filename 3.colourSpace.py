import cv2
from matplotlib import pyplot as plt
import numpy as np


def imshow(title="",image=None,size=7):
    h,w=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(image)
    plt.title(title)
    plt.show()

def channel_s(image,channel,r,g,b):
    zeros=np.zeros(image.shape[:2],dtype="uint8")
    match channel:
        case 1:return imshow("BLUE",cv2.merge([zeros,zeros,r]))

        case 2:return imshow("GREEN",cv2.merge([zeros,g,zeros]))
        
        case 3:return imshow("RED",cv2.merge([b,zeros,zeros]))

        case 4:return imshow("RED,BLUE",cv2.merge([b,zeros,r]))

        case 5:return imshow("BLUE,GREEN",cv2.merge([b,g,zeros]))

        case 6:return imshow("GREEN,RED",cv2.merge([zeros,g,r]))

        case 7:return imshow("RGB",cv2.merge([r,g,b]))

        case default:
            print("wrong input please select proper channel")
            return imshow("BLUE,GREEN,RED",cv2.merge([b,g,r]))

def boostcolour(image,r,g,b):
    boostvalue=int(input("enter the boost value: "))
    boostchannel=int(input("enter the channel that you want to boost: "))
    match boostchannel:
        case 1:return imshow(f"Red+{boostvalue}",cv2.merge([r+boostvalue,g,b]))

        case 2:return imshow(f"Blue+{boostvalue}",cv2.merge([r,g,b+boostvalue]))

        case 3:return imshow(f"Green+{boostvalue}",cv2.merge([r,g+boostvalue,b]))

        case default:return print("wrong input check the instruction")

def hue_sat_val(image,r,g,b):
    hueimage=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    imshow("Hue",hueimage)
    imshow("Hue", hueimage[:, :, 0])
    imshow("Saturation", hueimage[:, :, 1])
    imshow("Value", hueimage[:, :, 2])

def main():
    path='E:\Computer Vision stuff\OpenCV\images\images\castara.jpeg'
    image=cv2.imread(path)
    b,g,r=cv2.split(image)
    a=int(input('1-Blue channel\n2-Green channel\n3-Red channel\n4-RED,BLUE channel\n5-BLUE,GREEN channel\n6-GREEN,RED channel\n7-RGB\n'))
    channel_s(image,a,r,g,b)
    #boostcolour(image,r,g,b)
    #hue_sat_val(image,r,g,b)

main()