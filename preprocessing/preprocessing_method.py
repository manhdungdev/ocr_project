import cv2
import numpy as np
import pytesseract
from pytesseract import Output


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img1 = cv2.imread("../training/Images/page-book.jpg")
img2 = cv2.imread("../training/Images/eng_para_bw.jpg")
img3 = cv2.imread("../training/Images/recipe01.jpg")
img4 = cv2.imread("../training/Images/cup.jpg")
img5 = cv2.imread("../training/Images/book02.jpg")
img6 = cv2.imread("../training/Images/book_adaptative.jpg")
img7 = cv2.imread("../training/Images/img-process.jpg")
img8 = cv2.imread("../training/Images/text-opencv.jpg")
img9 = cv2.imread("../training/Images/text-opencv2.jpg")
img10 = cv2.imread("../training/Images/test_noise.jpg")
def simpleThreshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    value, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Simple Threshold", threshold)
    print(value)

    cv2.waitKey()

def otsuThreshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    otsuVal, otsuThreshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Otsu Threshold", otsuThreshold)
    print(otsuVal)
    cv2.waitKey()

def adaptiveThresholdAverage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    adaptive_average = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
    cv2.imshow("Adaptive Average", adaptive_average)
    cv2.waitKey()

def adaptiveThresholdGaussian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    adaptive_average = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
    cv2.imshow("Adaptive Average", adaptive_average)

    adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
    cv2.imshow("Adaptive Gaussian", adaptive_gaussian)
    cv2.waitKey()

def colorInversion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    print(gray)
    colorInversion = 255 - gray
    print(colorInversion)

    colorInversion = cv2.resize(colorInversion, None, fx = 1.5, fy = 1.5, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("Color Inversion", colorInversion)

    cv2.waitKey()

def erosion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    erosion = cv2.erode(gray, np.ones((5, 5), np.uint8))
    cv2.imshow("Erosion", erosion)
    cv2.waitKey()

def dilation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    dilation = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    cv2.imshow("Dilation", dilation)
    cv2.waitKey()

def opening(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    erosion = cv2.erode(gray, np.ones((5, 5), np.uint8))
    cv2.imshow("Erosion", erosion)

    dilation = cv2.dilate(erosion, np.ones((5, 5), np.uint8))
    cv2.imshow("Dilation", dilation)

    cv2.waitKey()

def closing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    dilation = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    cv2.imshow("Dilation", dilation)

    erosion = cv2.erode(dilation, np.ones((5, 5), np.uint8))
    cv2.imshow("Erosion", erosion)

    cv2.waitKey()

def noiseRemoving(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)

    #Average blur
    averageBlur = cv2.blur(gray, (5, 5))
    cv2.imshow("Average Blur", averageBlur)

    # Gaussian blur
    gaussianBlur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Gaussian Blur", gaussianBlur)

    # Median blur
    medianBlur = cv2.medianBlur(gray, 5)
    cv2.imshow("Median Blur", medianBlur)

    # Bilateral blur
    bilaterBlur = cv2.bilateralFilter(gray, 9, 50, 50)
    cv2.imshow("Bilateral Blur", bilaterBlur)

    cv2.waitKey()


# simpleThreshold(img1)
# simpleThreshold(img2)
# otsuThreshold(img3)
# adaptiveThresholdAverage(img5)
# adaptiveThresholdGaussian(img6)
# colorInversion(img7)
# erosion(img8)
# dilation(img9)
# opening(img8)
# closing(img9)
noiseRemoving(img10)
