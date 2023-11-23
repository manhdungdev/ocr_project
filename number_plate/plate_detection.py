import cv2
import numpy as np
import imutils
import pytesseract
from matplotlib import pyplot as plt
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
direction = "../training/Images/Images2"
images = [os.path.join(direction, f) for f in os.listdir(direction)]
print(images)

# img= cv2.imread("../training/Images/Images2/car3.jpg")
def showImg(img):
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def getText(img, plate, beginX, beginY, endX, endY):
    text = pytesseract.image_to_string(plate, lang="por+eng", config="--psm 6")
    text = text.replace(":", "-")
    text = "".join(ch for ch in text if ((ch.isalnum()) or ch == '-'))
    imgRel = cv2.putText(img, text, (beginX, beginY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 0), 2,
                         lineType=cv2.LINE_AA)
    imgRel = cv2.rectangle(img, (beginX, beginY), (endX, endY), (150, 255, 0), 2)
    showImg(imgRel)
    print(text)

def plate_detection(img):
    img = cv2.imread(img)
    (h, w) = img.shape[:2]
    # print(h, w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter( gray, 11, 17, 17)
    edge = cv2.Canny(blur, 30, 200)

    # cont = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cont = imutils.grab_contours(cont)
    # cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]
    cont = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)
    cont = sorted(cont, key = cv2.contourArea, reverse=True)[:6]


    location = None
    # Determine 4 countours bound the information
    for c in cont:
        perimeter = cv2.arcLength(c, True)
        appro = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if(cv2.isContourConvex(appro)):
            if (len(appro) == 4):
                location = appro
    # print(value)
    mask = np.zeros(gray.shape, np.uint8)
    imgPlate = cv2.drawContours(mask, [location], 0, 255, -1)
    img_Plate = cv2.bitwise_and(img, img, mask=mask)
    # showImg(img_Plate)

    (y, x) = np.where(mask== 255)
    (beginX, beginY) = (np.min(x), np.min(y))
    (endX, endY) = (np.max(x), np.max(y))
    plate = gray[beginY:endY, beginX:endX]
    # showImg(plate)

    getText(img, plate, beginX, beginY, endX, endY)

for image in images:
    plate_detection(image)

