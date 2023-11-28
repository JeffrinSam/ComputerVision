import cv2
from matplotlib import pyplot as plt


def imshow(title="",image=None,size=7):
    h,w=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    '''plt.imshow(image)'''
    plt.title(title)
    plt.show()

''' onlyafter coverting the image from BGR to RGB we can apply gray scale other wise it gives green dimenstional values'''

def grayscaler(image):
    gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray_image[0].shape)
    print(gray_image[0])
    imshow("castara,gray.jpeg",gray_image)   
    print(gray_image.shape)

def main():
    path='E:\Computer Vision stuff\OpenCV\images\images\castara.jpeg'
    image=cv2.imread(path)
    grayscaler(image) 

main()
