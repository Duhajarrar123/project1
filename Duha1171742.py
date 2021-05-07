import numpy as np
import cv2
import random
import os
ind=0
#imgsList=['book1_page11','moc_test_5','moc_test_6','moc_test_10','moc_train_1','moc_train_3','moc_train_15','moc_train_16','moc_train_17','moc_train_20']
imgsList=['book1_page19','book2_page2','book2_page5','book2_page6','moc_test_1','moc_test_2','moc_test_3','moc_test_4','moc_test_7','moc_test_8','moc_test_9','moc_train_2','moc_train_4','moc_train_5','moc_train_6','moc_train_9','moc_train_10','moc_train_11','moc_train_18','moc_train_19']

for ind in range(len(imgsList)):
    parent_dir = "C:/Users/duhaj/Desktop/ComputerVision/project1/OutputResults"
    path = os.path.join(parent_dir,imgsList[ind])
    os.mkdir(path)
    #pic = cv2.imread('ahte_dataset/ahte_test_binary_images/'+imgsList[ind]+'.png')
    pic = cv2.imread('ahte_dataset/ahte_train_binary_images/'+imgsList[ind]+'.png')
    imgcpy=pic.copy()
    # Creat a white image
    DILATE_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 100))
    whiteImg = cv2.morphologyEx(pic, cv2.MORPH_DILATE, DILATE_kernel, iterations=3)
    cv2.imwrite(path+'/whiteImg.png', whiteImg)
    # Convert image to grayScale
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path+"/GreyImg.png", gray)
    # Apply Threshold
    thresh = np.array(255 * (gray / 255) ** 1 , dtype='uint8')
    cv2.imwrite(path+"/greyImgCorrelation.png", thresh)
    ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
    cv2.imwrite(path+'/ThresholdOtsu.png',thresh)
    # Apply Horizontal Projection
    thresh[thresh == 0] = 0
    thresh[thresh == 255] = 1
    horizontal_projection = np.sum(thresh, axis=1)
    height=thresh.shape[0]
    width = thresh.shape[1]
    a=0
    b=0
    c=0
    yp=0
    # Change Scale of Horizontal Projection and Draw It's Result
    for row in range(height):
        if round(2*int(horizontal_projection[row]*width/height),-3)>0:
            if abs(yp-row) > 90:
               a=random.randint(0,255)
               b=random.randint(0,255)
               c=random.randint(0,255)
               yp = row
            cv2.line(imgcpy, (0,row), (round(9*int(horizontal_projection[row]*width/height),-3),row) ,(a, b,c), 5)
            cv2.line(whiteImg, (0, row), (round(9 * int(horizontal_projection[row] * width / height), -3), row), (a, b, c), 5)
            #print(int(horizontal_projection[row]*width/height))
    cv2.imwrite(path+"/copyImage.png", imgcpy)
    DILATE_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (600, 7))
    whiteImg = cv2.morphologyEx(whiteImg, cv2.MORPH_ERODE, DILATE_kernel, iterations=3)
    cv2.imwrite(path+"/whiteImgCon.png", whiteImg)
    # Merge The Original Image With Horizontal Projection Lines Result
    res = cv2.addWeighted(pic,0.7,whiteImg,0.3,0)
    cv2.imwrite(path+"/res_2021.png", res)
    cv2.imwrite("OutputResults/AllResults/"+imgsList[ind]+".png", res)

###########################################################################################################################
##################################################### My Drafts ############################################################
###########################################################################################################################

# import cv2
# import numpy as np
# #imgName='book1_page11.png'
# imgsList=['book1_page11.png','moc_test_5.png','moc_test_6.png','moc_test_10.png','moc_train_1.png','moc_train_3.png','moc_train_15.png','moc_train_16.png','moc_train_17.png','moc_train_20.png']
# flag=0
# for ind in range(0,1):
#     for dil in range(1,15):
#         for ero in range(1,15):
#             image = cv2.imread('ahte_dataset/ahte_test_binary_images/' + imgsList[ind])
#             cv2.imwrite('output/origin.png', image)
#             imgcpy = image.copy()
#
#             edges = cv2.Canny(image, 10, 10)
#             cv2.imwrite("output/canny.png", edges)
#
#             kernel = 3
#             image = cv2.medianBlur(image, kernel)
#             cv2.imwrite('output/median10.png', image)
#
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             cv2.imwrite("output/grey9.jpg", gray)
#
#             gray = np.array(255 * (gray / 255) ** 1, dtype='uint8')
#             cv2.imwrite("output/greyImgGamaCorrelation9.jpg", gray)
#
#             ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#             cv2.imwrite('output/otsus9.png', thresh)
#
#             cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#             for c in cnts:
#                 area = cv2.contourArea(c)
#                 if area < 100:
#                     cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
#             cv2.imwrite('output/nonoise.png', thresh)
#
#             ERODE_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero,1))
#             ERODE = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, ERODE_kernel, iterations=3)
#             cv2.imwrite('output/ERODE.png',ERODE)
#
#             dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,dil))
#             dilate = cv2.morphologyEx(ERODE, cv2.MORPH_DILATE, dilate_kernel, iterations=3)
#             cv2.imwrite('output/dilate.png',dilate)
#
#             close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (152,1))
#             close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, close_kernel, iterations=1)
#             cv2.imwrite('output/close9.png',close)
#             #close = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
#             k=50
#             cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#             s=0
#             yRound=[]
#             yperv=0
#             for c in cnts:
#                 area = cv2.contourArea(c)
#                 x, y, w, h = cv2.boundingRect(c)
#                 #print(area)
#                 #print("abs(w)= ",abs(w))
#                 if abs(w) >= 800 and round(y,-1) not in yRound:
#                     k+=40
#                     s += 1
#                     yperv=y
#                     yRound.append(round(y,-1))
#                     cv2.rectangle(imgcpy, (300, y), (2600, y + h), (k%255,(k+60)%255,(k*2+50)%255), 5)
#                     cv2.putText(imgcpy, text=str("l#"+str(s)), org=(100, y),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2, color=(k%255,(k+60)%255,(k*2+50)%255),thickness=3, lineType=cv2.LINE_AA)
#             if(s==27):
#                 print("reached :))))))) !")
#                 flag=1
#                 break
#             #else:
#                 print(ero,"-----",dil,"   s= ",s)
#         if flag==1:
#             cv2.imwrite('output/res/Duha'+str(ero)+str(dil)+imgsList[ind], imgcpy)
#             flag=0
#             break
#         #print("==================================")
#     #break
#
#     print("****** s= ",s," ******")
# # edges = cv2.Canny(thresh,1000,1000)
# # #cv2.imshow("Edge Detected Image", edges)
# # cv2.imwrite("output/canny9.png",edges)
# #
# # edges = cv2.Canny(thresh,10,10)
# # #cv2.imshow("Edge Detected Image", edges)
# # cv2.imwrite("output/canny99.png",edges)
# #
# # ret,thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU) #taskbek
# # cv2.imwrite('output/otsus9.png',thresh)
# #
# # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# # for c in cnts:
# #     area = cv2.contourArea(c)
# #     if area < 400:
# #         x,y,w,h = cv2.boundingRect(c)
# #         cv2.rectangle(thresh, (x, y), (x + w, y + h), (0,0,0), -1)
# # cv2.imwrite('output/threshImg9.png',thresh)
# #
# # blur = cv2.blur(thresh,(15,15))
# # cv2.imwrite('output/blureImg9.tif',blur)
# #
# # _,imgPart = cv2.threshold(thresh,1,255,cv2.THRESH_BINARY)
# # cv2.imwrite('output/imgPage9.png',imgPart)
# #
# # edges = cv2.Canny(imgPart,10,10)
# # #cv2.imshow("Edge Detected Image", edges)
# # cv2.imwrite("output/canny99.png",edges)
# #
# # (contoursIm, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # cv2.imwrite('output/imgpart9.png',imgPart)
# #
# #
# # for c in contoursIm:
# #     area = cv2.contourArea(c)
# #     cv2.drawContours(imgcpy,[c],-1,(255,0,255),2)
# #     x,y,w,h = cv2.boundingRect(c)
# #     # cv2.rectangle(imgcpy, (x, y), (x , y+100), (255,0,255), -1)
# #     cv2.putText(imgcpy,"",(x,y+100),cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),5)
# # cv2.imwrite("output/Contoure9.png", imgcpy)
# # #########################################################
# #
# #
#####################################################################################################################
# import cv2
# import numpy as np
#
# image = cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png')
# cv2.imwrite('output/origin.png',image)
# imgcpy = image.copy()
#
# image = cv2.GaussianBlur(image,(5,5), 1)
# cv2.imwrite('output/gaussian.png',image)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("output/grey.jpg", gray)
#
# gray = np.array(255 * (gray / 255) ** 1 , dtype='uint8')
# cv2.imwrite("output/greyImgGamaCorrelation.jpg", gray)
#
# ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
# cv2.imwrite('output/otsus.png',thresh)
#
# kernel = np.ones((1,5), np.uint8)
# kernel2 = np.ones((5,1), np.uint8)
# # use closing morph operation but fewer iterations than the letter then erode to narrow the image
# # temp_img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel2,iterations=2)
# line_img = cv2.erode(thresh,kernel,iterations=2)
# line_img = cv2.morphologyEx(line_img,cv2.MORPH_CLOSE,kernel2,iterations=2)
# #line_img = cv2.dilate(line_img,kernel2,iterations=3)
# cv2.imwrite('output/lineIMG.png',line_img)
#
#
# dist_transform = cv2.distanceTransform(line_img,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# kernel3 = np.ones((3,3),np.uint8)
# sure_bg = cv2.dilate(line_img,kernel3,iterations=3)
#
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
#
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
#
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
#
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
# markers = cv2.watershed(image,markers)
# image[markers == -1] = [255,0,0]
# cv2.imwrite("output.png",markers)
#
#
#
# blur = cv2.blur(line_img,(13,7))
# cv2.imwrite('output/blureImg.tif',blur)
#
# _,imgPart = cv2.threshold(blur,1,255,cv2.THRESH_BINARY)
# cv2.imwrite('output/imgPage.png',imgPart)
#
# close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
# close = cv2.morphologyEx(imgPart, cv2.MORPH_CLOSE, close_kernel, iterations=3)
# kernel = np.ones((1,25), np.uint8)
# close = cv2.erode(close,kernel,iterations=2)
# cv2.imwrite('output/close.png',close)
#######################################################################################################################
# import cv2
# import numpy as np
#
# image = cv2.imread('ahte_dataset/ahte_test_binary_images/moc_test_6.png')
# cv2.imwrite('output/origin.png',image)
# imgcpy = image.copy()
#
# image = cv2.GaussianBlur(image,(5,5), 1)
# cv2.imwrite('output/gaussian.png',image)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("output/grey.jpg", gray)
#
# gray = np.array(255 * (gray / 255) ** 1 , dtype='uint8')
# cv2.imwrite("output/greyImgGamaCorrelation.jpg", gray)
#
# ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
# cv2.imwrite('output/otsus.png',thresh)
#
#
# kernel = np.ones((1,5), np.uint8)
# kernel2 = np.ones((5,1), np.uint8)
# # use closing morph operation but fewer iterations than the letter then erode to narrow the image
# # temp_img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel2,iterations=2)
# line_img = cv2.erode(thresh,kernel,iterations=2)
# line_img = cv2.morphologyEx(line_img,cv2.MORPH_CLOSE,kernel2,iterations=2)
# #line_img = cv2.dilate(line_img,kernel2,iterations=3)
# cv2.imwrite('output/lineIMG.png',line_img)
#
# blur = cv2.blur(line_img,(13,7))
# cv2.imwrite('output/blureImg.png',blur)
#
# _,imgPart = cv2.threshold(blur,1,255,cv2.THRESH_BINARY)
# cv2.imwrite('output/imgPage.png',imgPart)
#
#
#
# close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
# close = cv2.morphologyEx(imgPart, cv2.MORPH_CLOSE, close_kernel, iterations=3)
# kernel = np.ones((1,25), np.uint8)
# close = cv2.erode(close,kernel,iterations=2)
# cv2.imwrite('output/close.png',close)
#
#
# cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     print(area)
#     if area < 100:
#         x,y,w,h = cv2.boundingRect(c)
#         cv2.rectangle(close, (x, y), (x + w, y + h), (0,0,0), -1)
# cv2.imwrite('output/threshImg.png',close)
#
# kernel = np.ones((1,25), np.uint8)
# close = cv2.erode(close,kernel,iterations=2)
# cv2.imwrite('output/close.png',close)
#
# (contours, _) = cv2.findContours(close, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# k=50
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area > 10000:
#         k+=30
#         x,y,w,h = cv2.boundingRect(cnt)
#         #cv2.rectangle(imgcpy,(x-1,y-5),(x+w,y+h),(k%255,(k+60)%255,(k*2+50)%255),5)
# cv2.imwrite("output/imgContoure.png", imgcpy)
#
#
# # im = cv2.imread('ahte_test_binary_images/book1_page11.png', cv2.IMREAD_GRAYSCALE)
# # GaussianFilter= cv2.GaussianBlur(im, (5,5), 0)
# # _, binarizedImage = cv2.threshold(GaussianFilter, 127, 255, cv2.THRESH_BINARY)
#
#
# close[close == 0] = 0
# close[close == 255] = 1
# horizontal_projection = np.sum(close, axis=1)
# print(horizontal_projection)
# height, width = close.shape
# print('width : ', width)
# print('height : ', height)
# blankImage = np.zeros((height, width,3), np.uint8)
# i=0
# j=0
# for row in range(height):
#     i+=140
#     j+=70
#     cv2.line(blankImage, (0,row), (int(horizontal_projection[row]*width/height),row), (i%255, j%255, (i+j+70)%255), 5)
# #print(blankImage)
# cv2.imwrite("output/blankImage.png", blankImage)
#
# #
# # se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
# # bg=cv2.morphologyEx(blankImage, cv2.MORPH_DILATE, se)
# # out_gray=cv2.divide(blankImage, bg, scale=255)
# #
# # cv2.imwrite('output/gray9.png',out_gray)
# #
#
# blankImage = cv2.Canny(blankImage,10,10)
# cv2.imwrite('output/ob.png',blankImage)
# lines = cv2.HoughLinesP(blankImage, 1, np.pi/180, 100, minLineLength=100, maxLineGap=300)
# #print("*****",lines)
# cv2.imwrite("output/b1.png", imgcpy)
# #i%255, j%255, (i+j+70)%255
# k=50
#
# xArray=[]
# yArray=[]
# prevY=0
# count=0
# prevX=0
#
# #xxArray=sorted(lines)
# # print(lines[0])
# # print(lines[1])
# # print(lines[2])
# print("len lines =",len(lines))
# pp=[]
# ponit=[]
# c=0
# cc=0
# k=50
# linesArraySorted=[]
# yDuha=[]
# for i in lines:
#     k+=50
#     print("%%%%",i[0])
#     if(round(i[0][1],-2) not in yDuha):
#        print("@@@@@",i[0])
#        linesArraySorted.append(i[0])
#        yDuha.append(round(i[0][1],-2))
#        cv2.rectangle(imgcpy, (x - 30, i[0][1] - 50), (x + w + 30, i[0][1] + 50), (k % 255, (k + 60) % 255, (k * 2 + 50) % 255), 3)
# print("len before :",len(linesArraySorted))
# linesArraySorted=sorted(linesArraySorted,reverse=True, key = lambda x: x[0])
# print(linesArraySorted)
# cv2.imwrite("output/blankDuha.png", imgcpy)
# print("000000000000000000000000000000000000000000000000")
# ready=[]
# for i in range(len(linesArraySorted)-1):
#     if abs(linesArraySorted[i][1]-linesArraySorted[i+1][1])>50 :
#         ready.append(linesArraySorted[i])
# print("len after :",len(yDuha))
# for line in lines:
#     c+=1
#     x1, y1, x2, y2 = line[0]
#     ###### save in array
#
#     minY=(y1+y2)//2
#     #print("||||||||||||||| ",minY)
#     #print("y1 = ",y1,"      | y2 = > ",y2)
#     #cv2.line(imgcpy, (x1, y1), (x2,y2), (255,0,0),1)
#     #print("yPrev - y =",abs(prevY-round(minY,-1)))
#     xArray.append(x2)
#     # print(pp)
#     # print(y1, "************")
#    # print(x2,"$$$")
#     if(c%10==0):
#     #    print("cuuuut")
#         cc+=1
#         #cv2.rectangle(imgcpy, (0, y1), (width, y1), (0,0, 255),1)
#     if  (x2>500):
#         yArray.append(minY)
#         ponit.append(line[0])
#
#         pp.append(round(y1,-1))
#         k += 50
#         count+=1
#         if (abs(prevY - y1) < 60 and prevY != 0):
#             y1 = prevY
#         #cv2.rectangle(imgcpy, (x-30 , y1-50), (x+w+30,y1+50), (k%255,(k+60)%255,(k*2+50)%255),3)
#
#         #print(round(y1/5) ," ++++++ ")
#         prevY = y1
#         prevX=round(x2,-1)
#
# #xArray = list(set(xArray))
# print(cc,"...................")
# print(len(ponit))
# print("count = ",count)
# cv2.imwrite("output/blank.png", imgcpy)
# ponit=sorted(ponit, key = lambda x: x[1])
# #print()
# print("**********************")
# pa=[]
# for i in lines:
#     pa.append(i[0])
#  #   print(i[0])
#
# pa=sorted(pa, key = lambda x: x[1])
# #print(pa)
# r=[]
# print(";;;;;;;;;;;;;;;;;;;")
# duha=[]
# for j in pa:
#     if round(j[1],-1) not in r:
#         r.append(round(j[1],-1))
#         duha.append(j)
#
# #print(r)
#
# yy1=[]
# yy2=[]
# xx1=[]
# xx2=[]
# for i in range(count):
#     xx1.append(ponit[i][0])
#     yy1.append(ponit[i][1])
#     xx2.append(ponit[i][2])
#     yy2.append(ponit[i][3])
#
# # print(yy2)
# k=50
# yp=0
# #print(duha)
# count=0
# for line in ready:
#         x1, y1, x2, y2 = line
#         minY = (y1 + y2) // 2
#         k += 50
#         count += 1
#         #print(abs(yp - y1),"Hiiiii1",y1," |||| ", yp)
#         if(abs(yp - y1)<60 and yp != 0):
#             y1=yp
#         #cv2.rectangle(imgcpy, (x-100 , y1-10), (x+w+70,y1+70), (k%255,(k+60)%255,(k*2+50)%255),5)
#         yp=y1
# cv2.imwrite("output/blank1997.png", imgcpy)
# print("c=",count)
#
#
#
# # yArray = [round(num, -2) for num in yArray]
# # yArray=list(set(yArray))
# # yArray=sorted(yArray)
# # for i in yArray:
# #     cv2.rectangle(imgcpy, (x - 30, i - 50), (x + w + 30, i + 50), (k % 255, (k + 60) % 255, (k * 2 + 50) % 255),3)
# #     print(i)
# #print(yArray)
# #print(len(yArray))
#
#
# # print("avg = ",sum(xArray)/len(xArray))
# # xArray = [round(num, -2) for num in xArray]
# # xArray=list(set(xArray))
# # xArray=sorted(xArray)
# # for i in range(len(xArray)-1):
# #     print("!! ** ",xArray[i]%10 , xArray[i+1]%10)
# #     if xArray[i]//100 == xArray[i+1]//100 :
# #         xArray.remove(xArray[i])
# #print(xArray)
# #print(len(xArray))
#
#
# # (contours, _) = cv2.findContours(close, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#
# # for c in contours:
# #     area = cv2.contourArea(c)
# #     if area > 0:
# #         cv2.drawContours(imgcpy,[c],-1,(0,0,0),5)
# #         x,y,w,h = cv2.boundingRect(c)
# #         # cv2.rectangle(imgcpy, (x, y), (x , y+100), (255,0,255), -1)
# #         cv2.putText(imgcpy,"",(x,y+100),cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),5)
# # cv2.imwrite("output/imgContoure2.png", imgcpy)
#
#
# #try 1
# # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# # ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
# # kernel = np.ones((1,5), np.uint8)
# # kernel2 = np.ones((5,1), np.uint8)
# # line_img = cv2.erode(thresh,kernel,iterations=2)
# # line_img = cv2.dilate(line_img,kernel2,iterations=1)
# # img = line_img.copy()
# # cnts = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# # for c in cnts:
# #     area = cv2.contourArea(c)
# #     if area > 2500:
# #         x,y,w,h = cv2.boundingRect(c)
# #         cv2.rectangle(line_img, (x, y), (x + w, y + h), (0,0,0), -1)
# # img = img - line_img
# # kernel = np.ones((1,15), np.uint8)
# # img = cv2.erode(img,kernel,iterations=2)
# # cv2.imwrite("output/erosion.png", img)
#
# #************************************************
#######################################################################################################
# # import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt
# # img= cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png')
# # img1=cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png')
# # #images=np.concatenate(img(img,img1),axis=1)
# # images=np.concatenate((img,img1),axis=1)
# # cv2.imshow("Images",images)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # gray_img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# # hist=cv2.calcHist(gray_img,[0],None,[256],[0,256])
# # hist1=cv2.calcHist(gray_img1,[0],None,[256],[0,256])
# # plt.subplot(121)
# # plt.title("Image1")
# # plt.xlabel('bins')
# # plt.ylabel("No of pixels")
# # plt.plot(hist)
# # plt.subplot(122)
# # plt.title("Image2")
# # plt.xlabel('bins')
# # plt.ylabel("No of pixels")
# # plt.plot(hist1)
# # plt.show()
# # gray_img_eqhist=cv2.equalizeHist(gray_img)
# # gray_img1_eqhist=cv2.equalizeHist(gray_img1)
# # hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
# # hist1=cv2.calcHist(gray_img1_eqhist,[0],None,[256],[0,256])
# # plt.subplot(121)
# # plt.plot(hist)
# # plt.subplot(122)
# # plt.plot(hist1)
# # plt.show()
# # eqhist_images=np.concatenate((gray_img_eqhist,gray_img1_eqhist),axis=1)
# # #cv2.imshow("Images",eqhist_images)
# # cv2.imwrite("eqh.png",eqhist_images)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# #
# # clahe=cv2.createCLAHE(clipLimit=40)
# # gray_img_clahe=clahe.apply(gray_img_eqhist)
# # gray_img1_clahe=clahe.apply(gray_img1_eqhist)
# # images=np.concatenate((gray_img_clahe,gray_img1_clahe),axis=1)
# # cv2.imshow("Images",images)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# #
# # th=80
# # max_val=255
# # ret, o1 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY)
# # cv2.putText(o1,"Thresh_Binary",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
# # ret, o2 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY_INV)
# # cv2.putText(o2,"Thresh_Binary_inv",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
# # ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
# # cv2.putText(o3,"Thresh_Tozero",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
# # ret, o4 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO_INV)
# # cv2.putText(o4,"Thresh_Tozero_inv",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
# # ret, o5 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TRUNC)
# # cv2.putText(o5,"Thresh_trunc",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
# # ret ,o6=  cv2.threshold(gray_img_clahe, th, max_val,  cv2.THRESH_OTSU)
# # cv2.putText(o6,"Thresh_OSTU",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
# #
# # final=np.concatenate((o1,o2,o3),axis=1)
# # final1=np.concatenate((o4,o5,o6),axis=1)
# #
# # cv2.imwrite("Image1.jpg",final)
# # cv2.imwrite("Image2.jpg",final1)
# #
# #
# # gray_image = cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png',0)
# # gray_image1 = cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png',0)
# # thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# # thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
# # thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)
# # thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)
# # thresh11 = cv2.adaptiveThreshold(gray_image1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# # thresh21 = cv2.adaptiveThreshold(gray_image1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
# # thresh31 = cv2.adaptiveThreshold(gray_image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,5 )
# # thresh41 = cv2.adaptiveThreshold(gray_image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
# #
# # final=np.concatenate((thresh1,thresh2,thresh3,thresh4),axis=1)
# # final1=np.concatenate((thresh11,thresh21,thresh31,thresh41),axis=1)
# # cv2.imwrite('rect.jpg',final)
# # cv2.imwrite('rect1.jpg',final1)
# #
# #
# # gray_image = cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png',0)
# # gray_image1 = cv2.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png',0)
# # ret,thresh1 = cv2.threshold(gray_image,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # ret,thresh2 = cv2.threshold(gray_image1,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# #
# # cv2.imwrite('rect.jpeg',np.concatenate((thresh1,thresh2),axis=1))
#
# import cv2 as cv
#
# #b,g,r = cv.split(src) # Split the color image into three channels b, g, r
# #Back2src = cv.merge([b,g,r]) #merge
# #Src[:,:,2] = 255 # Set the third channel to 255
#
# src = cv.imread('ahte_dataset/ahte_test_binary_images/book1_page11.png')
# cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
# #cv.imshow('input image', src)
# print('src.shape:', src.shape)
#
# b, g, r = cv.split(src)  # Display separately after splitting
# #cv.imshow('b', b)
# print('b.shape:', b.shape)
# #cv.imshow('g', g)
# print('g.shape:', g.shape)
# #cv.imshow('r', r)
#
# Back2src = cv.merge([b, g, r])  # merge
# #cv.imshow('back2src', Back2src)
# print('back2src.shape:', Back2src.shape)
#
# src[:, :, 2] = 255  # Set the third channel to 255
# #cv.imshow('changed src', src)
# cv.imwrite('pp.png',Back2src)
# cv.waitKey(0)
# cv.destroyAllWindows()