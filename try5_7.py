import cv2
import numpy as np
import measure

img = cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png')
cv2.imwrite('output/origin.png',img)
imgcpy = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/grey9.jpg", gray)

gray = np.array(255 * (gray / 255) ** 1 , dtype='uint8')
cv2.imwrite("output/greyImgGamaCorrelation9.jpg", gray)

#ret,thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU,11,10)
thresh =cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 3, 5)
cv2.imwrite('output/otsus9.png',thresh)

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 300:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
cv2.imwrite('output/nonoise.png', thresh)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
# gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
# cv2.imwrite('output/gradient.png',gradient)
# noise removal


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 300:
#         cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
cv2.imwrite('output/res/nonoise.png', thresh)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imwrite('output/res/opening.png',opening)
# sure background area
kernel = np.ones((1,40),np.uint8)
sure_bg = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 2)
cv2.imwrite('output/res/sure_bg.png',sure_bg)

kernel = np.ones((1,5),np.uint8)
sure_bg = cv2.morphologyEx(sure_bg,cv2.MORPH_ERODE,kernel, iterations = 2)
cv2.imwrite('output/res/sure_bg1.png',sure_bg)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
cv2.imwrite('output/res/dist_transform.png',dist_transform)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv2.imwrite('output/res/sure_fg.png',sure_fg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
cv2.imwrite('output/res/markers1.png',markers)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv2.imwrite('output/res/markers.png',markers)
cv2.imwrite('output/res/img.png',img)