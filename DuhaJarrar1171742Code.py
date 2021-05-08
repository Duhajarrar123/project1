import numpy as np
import cv2
import random
imgsList=['book1_page11.png','moc_test_5.png','moc_test_6.png','moc_test_10.png','moc_train_1.png','moc_train_3.png','moc_train_15.png','moc_train_16.png','moc_train_17.png','moc_train_20.png']

import cv2
pic = cv2.imread('ahte_dataset/ahte_train_binary_images/moc_train_19.png')

imgcpy=pic.copy()
# Creat a white image
DILATE_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 100))
whiteImg = cv2.morphologyEx(pic, cv2.MORPH_DILATE, DILATE_kernel, iterations=3)
cv2.imwrite('output/whiteImg.png', whiteImg)
# Convert image to grayScale
print("Original Image : ",pic)

gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/grey.png", gray)


print("Grayscale Image : ",gray[2000:][1000:])
# Apply Threshold
ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
cv2.imwrite('output/otsus.png',thresh)

print("Image with Thresh (Binary & Otsu) : ",thresh)
# Apply Horizontal Projection
thresh[thresh == 0] = 0
thresh[thresh == 255] = 1
print("Horizontal Projection Image : ",thresh)
horizontal_projection = np.sum(thresh, axis=1)
height=thresh.shape[0]
width = thresh.shape[1]
print("Horizontal Projection energy Image : ",horizontal_projection)
for row in range(height):
    cv2.line(imgcpy, (0, row), (int(horizontal_projection[row] * width / height), row), (0,255,0), 5)
cv2.imwrite('output/horizontal_projection.png',imgcpy)
a=0
b=0
c=0
num=0
yp=0
s=0
# Change Scale of Horizontal Projection and Draw It's Result
for row in range(height):
    if round(2*int(horizontal_projection[row]*width/height),-3)>0:
        if num == 0:
            s += 1
            cv2.putText(pic, text=str("Line"), org=(50, row+30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
            num=1
        if abs(yp-row) > 90:
           a=random.randint(0,255)
           b=random.randint(0,255)
           c=random.randint(0,255)
           yp = row
           num=0

        cv2.line(imgcpy, (0,row), (round(9*int(horizontal_projection[row]*width/height),-3),row) ,(a, b,c), 5)
        cv2.line(whiteImg, (0, row), (round(9 * int(horizontal_projection[row] * width / height), -3), row), (a, b, c), 5)
    # else:
    #     cv2.line(whiteImg, (0, row), (round(9 * int(horizontal_projection[row] * width / height), -3), row), (255,255,255),5)
cv2.putText(pic, text=str("This image have a "+str(s-1)+" Lines"), org=(700 , row - 200 ), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4,color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        #print(int(horizontal_projection[row]*width/height))
cv2.imwrite("output/copyImage.png", imgcpy)
cv2.imwrite("output/whiteImgCon.png", whiteImg)
#
ERODE_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (800, 7))
whiteImg = cv2.morphologyEx(whiteImg, cv2.MORPH_ERODE, ERODE_kernel, iterations=3)

cv2.imwrite("output/whiteImgCon1.png", whiteImg)
# Merge The Original Image With Horizontal Projection Lines Result
res = cv2.addWeighted(pic,0.7,whiteImg,0.3,0)
cv2.imwrite("output/resDuha.png", res)

