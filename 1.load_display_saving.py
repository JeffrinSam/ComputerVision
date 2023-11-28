import cv2
from matplotlib import pyplot as plt

def main () :
    path_q='E:\Computer Vision stuff\OpenCV\dog_1.webp'
    image=cv2.imread(path_q)
    imshow("my dog",image)
    cv2.imwrite('my dog_rgb.jpeg',image)
    print(f"height(row):{image.shape[0]},width(column):{image.shape[1]},depth(dimensionorchannel):{image.shape[2]}")
    imshow("my dog",image)



def imshow(title="",image=None):
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


main()
