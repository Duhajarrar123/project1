import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png')
imgcpy=img.copy()
# b,g,r = cv2.split(img)
# rgb_img = cv2.merge([r,g,b])
#
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# plt.subplot(121),plt.imshow(rgb_img)
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(thresh, 'gray')
# plt.title("Otus's binary threshold"), plt.xticks([]), plt.yticks([])
# plt.imsave("t.png",thresh)
# #plt.show()
#
# # noise removal
# kernel = np.ones((2,2),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)



#img = cv2.imread('coins.jpg')
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((5,1),np.uint8)
kernel1 = np.ones((1,40),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel1, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

plt.subplot(131),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(thresh, 'gray')
plt.title("Otus's binary threshold"), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(closing, 'gray')
plt.title("morphologyEx"), plt.xticks([]), plt.yticks([])
#plt.show()
plt.imsave("c.png",closing)
#plt.imsave("o.png",opening)
sure_bg = cv2.dilate(closing,kernel1,iterations=3)
plt.imsave("cc.png",closing)
#
#
# close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
# closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, close_kernel, iterations=3)
# kernel = np.ones((1,25), np.uint8)
# closing = cv2.erode(closing,kernel,iterations=2)
#
# cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 400:
#         x,y,w,h = cv2.boundingRect(c)
#         cv2.rectangle(closing, (x, y), (x + w, y + h), (0,0,0), -1)
# cv2.imwrite('threshImg.png',closing)
#
# (contours, _) = cv2.findContours(closing, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#
# for cnt in contours:
# 	x,y,w,h = cv2.boundingRect(cnt)
# 	cv2.rectangle(imgcpy,(x-1,y-5),(x+w,y+h),(0,255,0),5)
# cv2.imwrite("imgContoure.png", imgcpy)
#
# for c in contours:
#     area = cv2.contourArea(c)
#     if area > 0:
#         cv2.drawContours(imgcpy,[c],-1,(255,0,255),5)
#         x,y,w,h = cv2.boundingRect(c)
#         # cv2.rectangle(imgcpy, (x, y), (x , y+100), (255,0,255), -1)
#         cv2.putText(imgcpy,"",(x,y+100),cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),5)
# cv2.imwrite("imgContoure2.png", imgcpy)