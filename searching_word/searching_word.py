
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import os
import re
from matplotlib import pyplot as plt
from PIL import ImageFont, ImageDraw, Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
font = "training/Fonts/calibri.ttf"
direction = "../training/Images/Images1"
min_conf = 30
word_search = "computador"

images = [os.path.join(direction, f) for f in os.listdir(direction)]
print(images)

def writeText(text, x, y, img, font, color=(50, 50, 255), font_size = 32):
    font = ImageFont.truetype(font, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y - font_size), text, font = font, fill=color)
    img = np.array(img_pil)
    return img

def showImg(img):
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def boundingBox(result, img, i, color=(255, 100, 0)):
    x = result['left'][i]
    y = result['top'][i]
    w = result['width'][i]
    h = result['height'][i]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return x, y, img

def ocr_process(img, word_search, min_conf):
    result = pytesseract.image_to_data(img, lang="por", output_type=Output.DICT)
    for line in result:
        print(line, ":", result[line])
    numbers_of_times = 0
    for i in range(0, len(result["text"])):
        conf = int(result["conf"][i])
        if(conf > min_conf):
            text = result["text"][i]
            if(word_search.lower() in text.lower()):
                x, y, img = boundingBox(result, img, i, (0, 0, 255))
                img = writeText(text, x, y, img, font, (50, 50, 255), 14)
                numbers_of_times += 1
    return img, numbers_of_times



for image in images:
    img = cv2.imread(image)
    file_name = os.path.split(image)[-1]
    print("--------------\n" + file_name)

    img, numbers_of_times = ocr_process(img, word_search, min_conf)
    print("Number of times {0} in {1} is: {2}".format(word_search, file_name, numbers_of_times))
    print("\n")
    showImg(img)
