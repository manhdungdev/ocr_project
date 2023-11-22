import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import os
import re
from matplotlib import pyplot as plt
from PIL import ImageFont, ImageDraw, Image
from wordcloud import WordCloud

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
direction = "../training/Images/Images1"
images = [os.path.join(direction, f) for f in os.listdir(direction)]
print(images)



def showImg(img):
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def getText(img):
    text = pytesseract.image_to_string(img, lang="por")
    return text

fullText = ""
word_search = "computador"

for image in images:
    img = cv2.imread(image)
    file_name = os.path.split(image)[-1]
    fullText = fullText + file_name + "\n"
    text = getText(img)
    fullText = fullText + text
    # showImg(img)

# f = open(resultFile)
# result = [i.start() for i in re.finditer(word_search, f.read())]
# print(result)
# print(len(result))

for image in images:
    img = cv2.imread(image)
    file_name = os.path.split(image)[-1]
    print("-----------------\n" + file_name)
    text = getText(img)
    result = [i.start() for i in re.finditer(word_search, text)]
    print("Times in {0} is: {1}".format(word_search, len(result)))


