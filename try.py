import numpy as np
import cv2
import random
def turningpoints(x):
  N=0
  xArray=[]
  yArray=[]
  for i in range(1, len(x)-1):
     if (x[i-1] < x[i] and x[i+1] < x[i]):
         yArray.append(i)
         xArray.append(x[i])
         N+=1
  return N,xArray,yArray
imgsList=['book1_page11.png','moc_test_5.png','moc_test_6.png','moc_test_10.png','moc_train_1.png','moc_train_3.png','moc_train_15.png','moc_train_16.png','moc_train_17.png','moc_train_20.png']

pic = cv2.imread('ahte_dataset/ahte_test_binary_images/moc_train_20.png')
imgcpy=pic.copy()
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/grey.jpg", gray)

thresh = np.array(255 * (gray / 255) ** 1 , dtype='uint8')
cv2.imwrite("output/greyImgGamaCorrelation.jpg", thresh)

ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
cv2.imwrite('output/otsus.png',thresh)

ERODE_kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 100))
whiteImg = cv2.morphologyEx(pic, cv2.MORPH_DILATE, ERODE_kernel1, iterations=3)
cv2.imwrite('output/whiteImg.png', whiteImg)

# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 700:
#             cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
# cv2.imwrite('output/nonoise.png', thresh)
########################################################
# ERODE_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero, 1))
# ERODE = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, ERODE_kernel, iterations=3)
# cv2.imwrite('output/ERODE.png', ERODE)
#
# dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dil))
# dilate = cv2.morphologyEx(ERODE, cv2.MORPH_DILATE, dilate_kernel, iterations=3)
# cv2.imwrite('output/dilate.png', dilate)

# close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (170, 1))
# close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)
# cv2.imwrite('output/close9.png', close)
# thresh = close
#
# ERODE_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 13))
# ERODE = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, ERODE_kernel, iterations=3)
# cv2.imwrite('output/ERODE.png', ERODE)
# thresh=ERODE
thresh[thresh == 0] = 0
thresh[thresh == 255] = 1
horizontal_projection = np.sum(thresh, axis=1)
print(horizontal_projection)
height, width = thresh.shape
print('width : ', width)
print('height : ', height)
blankImage = np.zeros((height, width,3), np.uint8)
i=0
j=0
l=0
sumPro=0
xprofPro=[]
a=0
b=0
c=0
num=0
yp=0
for row in range(height):
    i+=140
    j+=70
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
        xprofPro.append(int(horizontal_projection[row]*width/height))

cv2.imwrite("output/blankImage.png", imgcpy)
cv2.imwrite("output/whiteImgCon.png", whiteImg)


res = cv2.addWeighted(pic,0.7,whiteImg,0.3,0)
cv2.imwrite("output/res3_2021.png", res)
print("num=",num)
ERODE_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,8))
ERODE = cv2.morphologyEx(imgcpy, cv2.MORPH_DILATE, ERODE_kernel, iterations=3)
cv2.imwrite('output/duhaERODE.png', ERODE)

ERODE_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
ERODE = cv2.morphologyEx(imgcpy, cv2.MORPH_DILATE, ERODE_kernel, iterations=3)
cv2.imwrite('output/duhaERODEtext.png', ERODE)

gray = cv2.cvtColor(ERODE, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/grey.jpg", gray)

thresh = np.array(255 * (gray / 255) ** 1 , dtype='uint8')
cv2.imwrite("output/greyImgGamaCorrelation.jpg", thresh)

ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
cv2.imwrite('output/otsus.png',thresh)

k=50
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts) == 2:
    cnts = cnts[0]
else:
    cnts[1]
s=0
for c in cnts:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    #print(area)
    #print("abs(w)= ",abs(w))
    # if abs(w) >= 800:
    k+=50
    s += 1
    #cv2.rectangle(imgcpy, (x, y-50), (x + w, y + 50), (k%255,(k+60)%255,(k*2+50)%255), 5)
        #cv2.putText(imgcpy, text=str("line #"+str(s)), org=(2100, y + h),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2, color=(k%255,(k+60)%255,(k*2+50)%255),thickness=3, lineType=cv2.LINE_AA)
cv2.imwrite('output/duhab1_1.png',imgcpy)
print("s=",s)




########################################################
print(xprofPro)
print(len(xprofPro))
c=0
val=0
yArray=[]
xArray=[]
while(c!=len(xprofPro)):
    if(xprofPro[c]>=val):
        val=xprofPro[c]
    else:
        yArray.append(c)
        xArray.append(val)
        val=0
    c+=1


n,xArray,yArray=turningpoints(xprofPro)
print(xArray)
print(yArray)
print(len(yArray),len(xArray))
roundContour=[]
k=50
z=0
yy=[]
for i in range(len(yArray)-1):
    print(abs(yArray[i]-yArray[i+1]))
    if(abs(yArray[i]-yArray[i+1])>=24):
        yy.append(yArray[i])
print('y=',len(yy))

for i in yy:
    #if round(i,-2) not in roundContour:
        z+=1
        k+=50
        roundContour.append(round(i,-2))
        cv2.rectangle(imgcpy, (400, i-20), (imgcpy.shape[0] - 400, i+100), (k % 255, (k + 60) % 255, (k * 2 + 50) % 255),5)
        cv2.putText(imgcpy, text=str("l#" + str(z)), org=(100, i+100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,color=(k % 255, (k + 60) % 255, (k * 2 + 50) % 255), thickness=3, lineType=cv2.LINE_AA)

print(len((roundContour)))
cv2.imwrite("output/projection.png", imgcpy)


# thresholdvalue1 = 50 #within this range all value will go to 0 , meaning black
# thresholdvalue2 = 100  #all value above going to set to 1 - white
# imgcpy=pic.copy()
# canny = cv2.Canny(pic,thresholdvalue1,thresholdvalue2)  # edges around the image
# cv2.imwrite('output/cFilter.png',canny)
#
# gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("output/grey9.jpg", gray)
#
# gray = np.array(255 * (gray / 255) ** 1, dtype='uint8')
# cv2.imwrite("output/greyImgGamaCorrelation9.jpg", gray)
#
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imwrite('output/otsus9.png', thresh)
# #
# # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
# # dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, dilate_kernel, iterations=3)
# # cv2.imwrite('output/dilate.png',dilate)
# #
# # ERODE_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
# # ERODE = cv2.morphologyEx(dilate, cv2.MORPH_ERODE, ERODE_kernel, iterations=3)
# # cv2.imwrite('output/ERODE.png',ERODE)
# #
#
#
# close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
# close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=3)
# cv2.imwrite('output/close9.png',close)
#
#
#
# ERODE_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,20))
# ERODE = cv2.morphologyEx(close, cv2.MORPH_ERODE, ERODE_kernel, iterations=3)
# cv2.imwrite('output/ERODEb.png',ERODE)
#
# blur = cv2.GaussianBlur(ERODE,(81,81),0)
# cv2.imwrite('output/blurTry.png',blur)
#
# cnts = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# s=0
# k=30
# for c in cnts:
#     area = cv2.contourArea(c)
#     x, y, w, h = cv2.boundingRect(c)
#     print(area)
#     # if abs(w) >= 800:
#     k+=50
#     s += 1
#     cv2.rectangle(imgcpy, (x, y), (x + w, y + h), (k%255,(k+60)%255,(k*2+50)%255), 5)
# cv2.imwrite('output/res1.png',imgcpy)
# print(s)



# import cv2 as cv
# import numpy as np
# import argparse
#
# W = 52  # window size is WxW
# C_Thr = 0.43  # threshold for coherency
# LowThr = 50  # threshold1 for orientation, it ranges from 0 to 180
# HighThr = 100  # threshold2 for orientation, it ranges from 0 to 180
#
#
# def calcGST(inputIMG, w):
#     img = inputIMG.astype(np.float32)
#     # GST components calculation (start)
#     # J =  (J11 J12; J12 J22) - GST
#     imgDiffX = cv.Sobel(img, cv.CV_32F, 1, 0, 3)
#     imgDiffY = cv.Sobel(img, cv.CV_32F, 0, 1, 3)
#     imgDiffXY = cv.multiply(imgDiffX, imgDiffY)
#
#     imgDiffXX = cv.multiply(imgDiffX, imgDiffX)
#     imgDiffYY = cv.multiply(imgDiffY, imgDiffY)
#     J11 = cv.boxFilter(imgDiffXX, cv.CV_32F, (w, w))
#     J22 = cv.boxFilter(imgDiffYY, cv.CV_32F, (w, w))
#     J12 = cv.boxFilter(imgDiffXY, cv.CV_32F, (w, w))
#     # GST components calculations (stop)
#     # eigenvalue calculation (start)
#     # lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
#     # lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
#     tmp1 = J11 + J22
#     tmp2 = J11 - J22
#     tmp2 = cv.multiply(tmp2, tmp2)
#     tmp3 = cv.multiply(J12, J12)
#     tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
#     lambda1 = 0.5 * (tmp1 + tmp4)  # biggest eigenvalue
#     lambda2 = 0.5 * (tmp1 - tmp4)  # smallest eigenvalue
#     # eigenvalue calculation (stop)
#     # Coherency calculation (start)
#     # Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
#     # Coherency is anisotropy degree (consistency of local orientation)
#     imgCoherencyOut = cv.divide(lambda1 - lambda2, lambda1 + lambda2)
#     # Coherency calculation (stop)
#     # orientation angle calculation (start)
#     # tan(2*Alpha) = 2*J12/(J22 - J11)
#     # Alpha = 0.5 atan2(2*J12/(J22 - J11))
#     imgOrientationOut = cv.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)
#     imgOrientationOut = 0.5 * imgOrientationOut
#     # orientation angle calculation (stop)
#     return imgCoherencyOut, imgOrientationOut
#
#
#
# imgIn = cv.imread('ahte_dataset/ahte_test_binary_images/moc_train_1.png')
#
# imgCoherency, imgOrientation = calcGST(imgIn, W)
# _, imgCoherencyBin = cv.threshold(imgCoherency, C_Thr, 255, cv.THRESH_BINARY)
# _, imgOrientationBin = cv.threshold(imgOrientation, LowThr, HighThr, cv.THRESH_BINARY)
# imgBin = cv.bitwise_and(imgCoherencyBin, imgOrientationBin)
# imgCoherency = cv.normalize(imgCoherency, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# imgOrientation = cv.normalize(imgOrientation, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# cv.imwrite('result.jpg', np.uint8(0.5 * (imgIn + imgBin)))
# cv.imwrite('Coherency.jpg', imgCoherency)
# cv.imwrite('Orientation.jpg', imgOrientation)
