import cv2 #Thư viện OpenCV dùng cho xử lý hình ảnh.
import numpy as np
import pytesseract #Thư viện dùng để nhận diện văn bản trong hình ảnh.
from pytesseract import Output


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Hàm này vẽ một hình chữ nhật xung quanh văn bản được nhận diện
def boundingBox(result, img, i, color=(255, 100, 0)): 
    x = result['left'][i]
    y = result['top'][i]
    w = result['width'][i]
    h = result['height'][i]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return x, y, img

#Đọc hình ảnh từ đường dẫn chỉ định.
img = cv2.imread("training/Images/test01.jpg")

#Chuyển hình ảnh sang không gian màu RGB
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



