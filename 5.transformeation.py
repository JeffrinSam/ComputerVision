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


def Wrap_affin(image,n):
    # Our Translation
    #       | 1 0 Tx |
    #  T  = | 0 1 Ty |
    h,w,d=image.shape[:]
    print(h,w,d)
    Tx,Ty=h/n,w/n
    T=np.float32([[1,0,Tx],[0,1,Ty]])
    print('Translation',T)
    Translation_image=cv2.warpAffine(image,T,(w,h))
    imshow('Translation',Translation_image)
    #syntax;cv2.warpAffine(image, T, (width, height)) warpaffine

def Rotational_affin(image,n,scale):
    #cv2.getRotationMatrix2D(rotation_center_x, rotation_center_y, angle of rotation, scale)
    # Our Rotaionmatrix
    #       | cosa -sina |
    #  M  = | sina  cosa |
    h,w,d=image.shape[:]
    rotatedmatrix=cv2.getRotationMatrix2D((w/2,h/2),n,scale)
    rotat_image=cv2.warpAffine(image,rotatedmatrix,(w,h))
    imshow("rotatedmatrix",rotat_image)







def main():
    image=cv2.imread('E:\Computer Vision stuff\OpenCV\dog_1.webp')
    imshow('orginal',image)
    option=int(input('SELECT the Options \n1.Transilate Image\n2.Rotate Image\n3.Transpose Image\n'))
    match option:
        case 1:
            n=int(input('Enter the Wrap Affin value(1,2,4,8,..,2**) :'))
            Wrap_affin(image,n)
        case 2:
            angle=int(input("Enter the Angle of rotation(0,30,45,60,90-360) :"))
            scale=float(input("Enter the zoom (0.25,0.5,1,2,3) :"))
            Rotational_affin(image,angle,scale)
        case 3:
            Tans=cv2.transpose(image)
            imshow('transpose',Tans)
        case default:prinT('SELECT 1 2 3 WRONG ENTRY')
    imshow('orginal',image)

main()
    
'''Syntax: cv2.flip(src, flipCode, dst )
Parameters: src: Input array.
dst: Output array of the same size and type as src.
flip code: A flag to specify how to flip the array; 
0 means flipping around the x-axis and positive value 
(for example, 1) means flipping around y-axis. Negative value (for example, -1)
 means flipping around both axes. Return Value: It returns an image. '''   
