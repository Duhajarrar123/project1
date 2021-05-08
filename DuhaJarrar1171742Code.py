import numpy as np
import cv2
import random
imgsList=['book1_page11.png','moc_test_5.png','moc_test_6.png','moc_test_10.png','moc_train_1.png','moc_train_3.png','moc_train_15.png','moc_train_16.png','moc_train_17.png','moc_train_20.png']
pic = cv2.imread('ahte_dataset/ahte_test_binary_images/moc_train_20.png')
imgcpy=pic.copy()
# Creat a white image
DILATE_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 100))
whiteImg = cv2.morphologyEx(pic, cv2.MORPH_DILATE, DILATE_kernel, iterations=3)
cv2.imwrite('output/whiteImg.png', whiteImg)
# Convert image to grayScale
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/grey.png", gray)
# Apply Threshold
#thresh = np.array(255 * (gray / 255) ** 1 , dtype='uint8')
#cv2.imwrite("output/greyImgGamaCorrelation.png", thresh)
ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
cv2.imwrite('output/otsus.png',thresh)
# Apply Horizontal Projection
thresh[thresh == 0] = 0
thresh[thresh == 255] = 1
horizontal_projection = np.sum(thresh, axis=1)
height=thresh.shape[0]
width = thresh.shape[1]
a=0
b=0
c=0
num=0
yp=0
# Change Scale of Horizontal Projection and Draw It's Result
for row in range(height):
    if round(2*int(horizontal_projection[row]*width/height),-3)>0:
        num+=1
        if abs(yp-row) > 90:
           a=random.randint(0,255)
           b=random.randint(0,255)
           c=random.randint(0,255)
           yp = row
        cv2.line(imgcpy, (0,row), (round(9*int(horizontal_projection[row]*width/height),-3),row) ,(a, b,c), 5)
        cv2.line(whiteImg, (0, row), (round(9 * int(horizontal_projection[row] * width / height), -3), row), (a, b, c), 5)
        #print(int(horizontal_projection[row]*width/height))
cv2.imwrite("output/copyImage.png", imgcpy)
cv2.imwrite("output/whiteImgCon.png", whiteImg)
dilImg=whiteImg
DILATE_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (600, 7))
dilImg = cv2.morphologyEx(dilImg, cv2.MORPH_ERODE, DILATE_kernel, iterations=3)
cv2.imwrite('output/dilImg.png', dilImg)
resD = cv2.addWeighted(pic,0.7,dilImg,0.3,0)
cv2.imwrite("output/resDil_2021.png", resD)

# Merge The Original Image With Horizontal Projection Lines Result
res = cv2.addWeighted(pic,0.7,whiteImg,0.3,0)
cv2.imwrite("output/res85_2021.png", res)

