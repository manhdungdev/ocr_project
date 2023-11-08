import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import ImageFont, ImageDraw, Image



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
font = "training/Fonts/arial.ttf"

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
    draw.text((x, y - 10), text, font = font)
    img = np.array(img_pil)
    return img

img = cv2.imread("training/Images/test02-02.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Img", rgb)
result = pytesseract.image_to_data(rgb, lang="por", output_type=Output.DICT)
print(result)
for line in result:
    print(line, ":", result[line])

img_copy = rgb.copy()
for i in range(0, len(result['text'])):
    if(result['conf'][i] > 40):
        print(result['conf'][i])
        x, y, img = boundingBox(result, img_copy, i)
        text = result['text'][i]
        # cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),2, cv2.LINE_8)
        img_copy = writeText(text, x, y, img_copy, font)
imS = cv2.resize(img_copy, (960, 540))
cv2.imshow("Img", imS)
cv2.waitKey()