import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open("training/Images/book01.jpg")
plt.imshow(img)
plt.show()
osd = pytesseract.image_to_osd(img, config='--psm 0 -c min_characters_to_try=5')
print(osd)