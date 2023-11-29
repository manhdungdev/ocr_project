import cv2
import numpy as np
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("training/Images/img_preprocessing1.jpg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Original Img", rgb)

transform1 = cv2.resize(rgb.copy(), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Img x 1.5", transform1)

transform2 = cv2.resize(rgb.copy(), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Img / 1.5", transform2)
cv2.waitKey()
