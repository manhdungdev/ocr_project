import cv2
import numpy as np
import pytesseract
from pytesseract import Output


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def boundingBox(result, img, i, color=(255, 100, 0)):
    x = result['left'][i]
    y = result['top'][i]
    w = result['width'][i]
    h = result['height'][i]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return x, y, img

img = cv2.imread("training/Images/test01.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Img", rgb)
result = pytesseract.image_to_data(rgb, lang="eng", output_type=Output.DICT)
print(result)
for line in result:
    print(line, ":", result[line])

img_copy = rgb.copy()
for i in range(0, len(result['text'])):
    if(result['conf'][i] > 40):
        print(result['conf'][i])
        x, y, img = boundingBox(result, img_copy, i)
        text = result['text'][i]
        cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),2, cv2.LINE_8)
imS = cv2.resize(img_copy, (960, 540))
cv2.imshow("Img", imS)
cv2.waitKey()



