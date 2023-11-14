import cv2 
from matplotlib import pyplot as plt 
import numpy as np

def display (im_path):
    dpi = 80 
    im_data = plt.imread(im_path)

    height, width, depth = im_data.shape

    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')

    ax.imshow(im_data, cmap='gray')

    plt.show()

#new = cv2.imread("training/Images/page_01_rotated.jpg")
new = cv2.imread("training/Images/eng_para.jpg")

def getSkewAngle(cvImage) -> float:
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect 
        cv2.rectangle(newImage, (x,y), (x+w,y+h), (0,255,0),2)

    largestContour = contours[0]
    #print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("training/temp/boxes.jpg", newImage)
    
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

#Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

#Deskew image:
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    # Add a check to see if the angle is small enough to ignore
    if angle <= -90:
        return cvImage
    else:
        return rotateImage(cvImage, -1.0 * angle)

fixed = deskew(new)
print(getSkewAngle(new))
cv2.imwrite("training/temp/rotated_fixed.jpg", fixed)

display("training/temp/rotated_fixed.jpg")
