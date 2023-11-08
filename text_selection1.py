import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import ImageFont, ImageDraw, Image

def boundingBox(result, img, i, color=(255, 100, 0)):
    x = result['left'][i]
    y = result['top'][i]
    w = result['width'][i]
    h = result['height'][i]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return x, y, img

def writeText(text, x, y, img, font, font_size = 32):
    font = ImageFont.truetype(font, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y - font_size), text, font = font)
    img = np.array(img_pil)
    return img
def textDetectionPattern(img, pattern):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Img", rgb)
    result = pytesseract.image_to_data(rgb, lang="por", output_type=Output.DICT)
    print(result)
    for line in result:
        print(line, ":", result[line])

    img_copy = rgb.copy()
    dates = []
    for i in range(0, len(result['text'])):
        if(result['conf'][i] > 40):
            text = result['text'][i]
            if(re.match(pattern, text)):
                x, y, img = boundingBox(result, img_copy, i, (0, 0, 255))
                # cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),2, cv2.LINE_8)
                img_copy = writeText(text, x, y, img_copy, font, 12)
                dates.append(text)
            else:
                x, y, img = boundingBox(result, img_copy, i, )
    # img_copy = cv2.resize(img_copy, (960, 540))
    print(dates)
    cv2.imshow("Img", img_copy)
    cv2.waitKey()
def textDetection(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Img", rgb)
    result = pytesseract.image_to_data(rgb, lang="eng", output_type=Output.DICT)
    print(result)
    for line in result:
        print(line, ":", result[line])

    img_copy = rgb.copy()
    for i in range(0, len(result['text'])):
        if(result['conf'][i] > 40):
            text = result['text'][i]
            x, y, img = boundingBox(result, img_copy, i)
            cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),2, cv2.LINE_8)
    # img_copy = cv2.resize(img_copy, (960, 540))
    cv2.imshow("Img", img_copy)
    cv2.waitKey()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
font = "training/Fonts/arial.ttf"
datePattern = "^(0?[1-9]|[12][0-9]|3[01])[\/\-](0?[1-9]|1[012])[\/\-]\d{4}$"

img1 = cv2.imread("training/Images/table_test.jpg")
img2 = cv2.imread("training/Images/cup.jpg")


# textDetectionPattern(img1, datePattern)
textDetection(img2)