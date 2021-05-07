import cv2
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import tensorflow as tf
import tensorflow as tf
import tensorflow as tf

tf.compat.v1.disable_eager_execution()



import os
#print(os.listdir("../input"))
import warnings
#warnings.filterwarnings('ignore')
#%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
sns.set(rc={'figure.figsize' : (22, 10)})
sns.set_style("darkgrid", {'axes.grid' : True})

def showImg(img, cmap=None):
    plt.imshow(img, cmap=cmap, interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()

#############################
image = cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png')
cv2.imwrite('origin.png',image)
imgcpy = image.copy()

image = cv2.GaussianBlur(image,(5,5), 1) 
cv2.imwrite('gaussian.png',image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grey.jpg", gray)

gray = np.array(255 * (gray / 255) ** 1 , dtype='uint8')
cv2.imwrite("greyImgGamaCorrelation.jpg", gray)

ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
cv2.imwrite('otsus.png',thresh)

kernel = np.ones((1,5), np.uint8)
kernel2 = np.ones((5,1), np.uint8)	
# use closing morph operation but fewer iterations than the letter then erode to narrow the image	
# temp_img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel2,iterations=2)
line_img = cv2.erode(thresh,kernel,iterations=2)	
line_img = cv2.morphologyEx(line_img,cv2.MORPH_CLOSE,kernel2,iterations=2)
#line_img = cv2.dilate(line_img,kernel2,iterations=3)
cv2.imwrite('lineIMG.png',line_img)

blur = cv2.blur(line_img,(13,7))
cv2.imwrite('blureImg.tif',blur)

_,imgPart = cv2.threshold(blur,1,255,cv2.THRESH_BINARY)
cv2.imwrite('imgPage.png',imgPart)

close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
close = cv2.morphologyEx(imgPart, cv2.MORPH_CLOSE, close_kernel, iterations=3)
kernel = np.ones((1,25), np.uint8)
close = cv2.erode(close,kernel,iterations=2)	
cv2.imwrite('close.png',close)


cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 400:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(close, (x, y), (x + w, y + h), (0,0,0), -1)
cv2.imwrite('threshImg.png',close)

(contours, _) = cv2.findContours(close, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(imgcpy,(x-1,y-5),(x+w,y+h),(0,255,0),5)
cv2.imwrite("imgContoure.png", imgcpy)

for c in contours:
    area = cv2.contourArea(c)
    if area > 0:
        cv2.drawContours(imgcpy,[c],-1,(255,0,255),5)
        x,y,w,h = cv2.boundingRect(c)
        # cv2.rectangle(imgcpy, (x, y), (x , y+100), (255,0,255), -1)
        cv2.putText(imgcpy,"",(x,y+100),cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),5)
cv2.imwrite("imgContoure2.png", imgcpy)


#############################
img1 = cv2.imread('lineIMG.png')
#showImg(img1, cmap='gray')
print(img1.ndim)
print(img1.shape)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print(img2.shape)
#showImg(img2, cmap='gray')
img3 = np.transpose(img2)
#showImg(img3, cmap='gray')
img = np.arange(16).reshape((4,4))
img
#showImg(img, cmap='gray')
def createKernel(kernelSize, sigma, theta):
    "create anisotropic filter kernel according to given parameters"
    assert kernelSize % 2 # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel
kernelSize=9
sigma=4
theta=1.5
imgFiltered1 = cv2.filter2D(img3, -1, createKernel(kernelSize, sigma, theta), borderType=cv2.BORDER_REPLICATE)
#showImg(imgFiltered1, cmap='gray')
def applySummFunctin(img):
    res = np.sum(img, axis = 0)    #  summ elements in columns
    return res
def normalize(img):
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img
img4 = normalize(imgFiltered1)
(m, s) = cv2.meanStdDev(imgFiltered1)
m[0][0]
summ = applySummFunctin(img4)
print(summ.ndim)
print(summ.shape)
plt.plot(summ)
#plt.show()

def smooth(x, window_len=11, window='hanning'):
#     if x.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y

windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
smoothed = smooth(summ, 35)
plt.plot(smoothed)
#plt.show()

from scipy.signal import argrelmin
mins = argrelmin(smoothed, order=2)
arr_mins = np.array(mins)

plt.plot(smoothed)
plt.plot(arr_mins, smoothed[arr_mins], "x")
#plt.show()


def crop_text_to_lines(text, blanks):
    x1 = 0
    y = 0
    lines = []
    for i, blank in enumerate(blanks):
        x2 = blank
        print("x1=", x1, ", x2=", x2, ", Diff= ", x2 - x1)
        line = text[:, x1:x2]
        lines.append(line)
        x1 = blank
    return lines


def display_lines(lines_arr, orient='vertical'):
    plt.figure(figsize=(30, 30))
    if not orient in ['vertical', 'horizontal']:
        raise ValueError("Orientation is on of 'vertical', 'horizontal', defaul = 'vertical'")
    if orient == 'vertical':
        for i, l in enumerate(lines_arr):
            line = l
            #plt.subplot(2, 10, i+1 )  # A grid of 2 rows x 10 columns
            plt.axis('off')
            plt.title("Line #{0}".format(i))
           # _ = plt.imshow(line, cmap='gray', interpolation='bicubic')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    else:
        path=["l1","l2","l3","l4","l5"]
        k=0
        for i, l in enumerate(lines_arr):
            line = l
   #         plt.subplot(40, 1, i + 1)  # A grid of 40 rows x 1 columns
            plt.axis('off')
            plt.title("Line #{0}".format(i))
           # _ = plt.imshow(line, cmap='gray', interpolation='bicubic')
            plt.imsave("Line"+str(i)+".png",line)
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            k+=1
    #plt.show()

found_lines = crop_text_to_lines(img3, arr_mins[0])

#sess = tf.Session()
sess = tf.compat.v1.Session()
found_lines_arr = []
with sess.as_default():
    for i in range(len(found_lines)-1):
        found_lines_arr.append(tf.expand_dims(found_lines[i], -1).eval())

display_lines(found_lines)


def transpose_lines(lines):
    res = []
    for l in lines:
        line = np.transpose(l)
        res.append(line)
    return res

res_lines = transpose_lines(found_lines)
display_lines(res_lines, 'horizontal')
