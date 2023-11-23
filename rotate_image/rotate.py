import numpy as np
import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
imgO= cv2.imread("../training/Images/rotate_img.jpg")


#Determine all countours
def find_contour(img):
    cont = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)
    cont = sorted(cont, key = cv2.contourArea, reverse=True)[:6]
    return cont

def sortPoints(points):
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), dtype=np.int32)
    # print(newPoints)
    add = points.sum(1)
    # print(add)

    newPoints[0] = points[np.argmin(add)]
    newPoints[2] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    # print(diff)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[3] = points[np.argmax(diff)]
    # print(newPoints)

    return newPoints

def rotateImg(imgO):
    (H, W) = imgO.shape[:2]
    imgO = cv2.resize(imgO, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img = imgO.copy()

    # Get real width, height
    (h, w) = img.shape[:2]

    value = 0

    #Preprocessing image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # showImg(gray)
    # cv2.imshow("Gray", gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # showImg(gray)
    # cv2.imshow("Blur", blur)

    #Detect border of text
    edge = cv2.Canny(blur, 60, 160)
    # showImg(edge)
    # cv2.imshow("edge", edge)

    cont = find_contour(edge)

    #Determine 4 countours bound the information
    for c in cont:
        perimeter = cv2.arcLength(c, True)
        appro = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if(len(appro) == 4):
            value = appro
    # print(value)

    #Draw contour
    cv2.drawContours(img, value, -1, (120, 255, 0), 14)
    cv2.drawContours(img, [value], -1, (120, 255, 0), 2)
    cv2.imshow("Img contour", img)

    #Sort contours in order
    pointsValue = sortPoints(value)
    pts1 = np.float32(pointsValue)
    # print(pts1)
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # print(pts2)

    #Get the different perspective between 2 matrixes
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # print(matrix)

    transform = cv2.warpPerspective(imgO, matrix, (w, h))
    cv2.imshow("Transform", transform)

    #Processing image
    transform = cv2.resize(transform, None, fx = 1.5, fy = 1.5, interpolation=cv2.INTER_CUBIC)

    #Thresholding
    processedImg = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
    processedImg = cv2.adaptiveThreshold(processedImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
    # cv2.imshow("Image after applying thresholding", processedImg)

    #Opening image
    erosion = cv2.erode(processedImg, np.ones((3, 3), np.uint8))
    # cv2.imshow("Erosion", erosion)

    dilation = cv2.dilate(erosion, np.ones((3, 3), np.uint8))
    # cv2.imshow("Dilation", dilation)

    #Inversion image
    colorInversion = 255 - dilation
    colorInversion = cv2.resize(colorInversion, None, fx = 0.75, fy = 0.75, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("Color Inversion", colorInversion)


    cv2.waitKey()

rotateImg(imgO)