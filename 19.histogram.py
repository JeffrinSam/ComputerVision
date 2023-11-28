import cv2
import numpy as np
from matplotlib import pyplot as plt

path="E:/Computer Vision stuff/OpenCV/images/images/input.jpg"
image=cv2.imread(path)


def imshow(title="",image=None,size=7):
    w,h=image.shape[0],image.shape[1]
    aspectratio=w/h
    plt.figure(figsize=(size*aspectratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def hist(path):
    image = cv2.imread(path)
    imshow("Input", image)

    # histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    # We plot a histogram, ravel() flatens our image array
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

    # Viewing Separate Color Channels
    color = ('b', 'g', 'r')

    # We now separate the colors and plot each in the Histogram
    for i, col in enumerate(color):
        histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram2, color = col)
        plt.xlim([0,256])

    plt.show()
'''............................................................'''
def hist2():
    image = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/tobago.jpg')
    imshow("Input", image)

    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    # We plot a histogram, ravel() flatens our image array
    plt.hist(image.ravel(), 256, [0, 256]); plt.show()

    # Viewing Separate Color Channels
    color = ('b', 'g', 'r')

    # We now separate the colors and plot each in the Histogram
    for i, col in enumerate(color):
        histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram2, color = col)
        plt.xlim([0,256])

    plt.show()
'''............................................................'''
def centroidHistogram(clt):
    # Create a histrogram for the clusters based on the pixels in each cluster
    # Get the labels for each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

    # Create our histogram
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, so that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plotColors(hist, centroids):
    # Create our blank barchart
    bar = np.zeros((100, 500, 3), dtype = "uint8")

    x_start = 0
    # iterate over the percentage and dominant color of each cluster
    for (percent, color) in zip(hist, centroids):
      # plot the relative percentage of each cluster
      end = x_start + (percent * 500)
      cv2.rectangle(bar, (int(x_start), 0), (int(end), 100),
        color.astype("uint8").tolist(), -1)
      x_start = end
    return bar

'''..............................................................'''
from sklearn.cluster import KMeans

image = cv2.imread('E:/Computer Vision stuff/OpenCV/images/images/tobago.jpg')
imshow("Input", image)

# We reshape our image into a list of RGB pixels
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
image = image.reshape((image.shape[0] * image.shape[1], 3))
print(image.shape)

number_of_clusters = 5
clt = KMeans(number_of_clusters)
clt.fit(image)

hist = centroidHistogram(clt)
bar = plotColors(hist, clt.cluster_centers_)

# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()