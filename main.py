import cv2
import numpy as np
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("training/Images/exit.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Img", rgb)
text = pytesseract.image_to_string(rgb, lang="por", config="--psm 7")
print(text)
cv2.waitKey()

