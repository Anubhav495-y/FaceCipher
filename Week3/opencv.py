import cv2 as cv

# Read in an image
img = cv.imread('dog.jpg')
print(img.shape)
#cv.imshow('Dog', img)
#cv.waitKey(0)

# gray scaling
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print(gray.shape)

#RGB colors 
import numpy as np
Blue = img[:,:,0]
Green = img[:,:,1]
Red = img[:,:,2]

new_img = np.hstack((Blue,Green,Red))
#cv.imshow("separated",new_img)
#cv.waitKey(0)

#HSV
 
#resising
img_resized = cv.resize(img,(256,256))
print(img_resized.shape)
#cv.imshow("img",img_resized)
#cv.waitKey(0)

#flip
img_flip = cv.flip(img,0)  #(image, axis)

#croping
img_crop = img[50:200,200:326]

#rotate
height , width= img.shape[0],img.shape[1]

T = cv.getRotationMatrix2D((width/2,height/2),180,1)
img_rotate = cv.warpAffine(img,T,(width,height))

#Drawing shapes
image = np.zeros((512,512,3))

rect = cv.rectangle(image,pt1=(100,100),pt2=(300,300),color=(255,0,0),thickness=2)
circle = cv.circle(image,center=(100,400),radius=50,color=(0,255,0),thickness=-1)
line = cv.line(image,pt1=(100,100),pt2=(300,300),color=(255,0,0),thickness=2)
text = cv.putText(image,org=(400,400),fontScale=4,color=(0,0,255),thickness=2,lineType=cv.LINE_AA,text="HEY!!",fontFace=cv.FONT_HERSHEY_SCRIPT_COMPLEX)


#cv.imshow("shapes",image)
#cv.waitKey(0)

#Blurring
  
  #Average Blur 
avg = cv.blur(img,(7,7))  #(image,Kernel size) you average the pixel intensity of kernel centre with neigbouring pixel intensities
#cv.imshow("Blurred",avg)

  #Gaussian Blur 
gauss = cv.GaussianBlur(img,(7,7),0)  
#cv.imshow("gaussian Blurr",gauss)

  #median Blur
median = cv.medianBlur(img, 3 ) 
#cv.imshow("meadian blur",median)

  #Bilateral blurr
bil = cv.bilateralFilter(img,5,15,10)  
#cv.imshow("bilateral",bil)

#cv.imshow("OG",img)
#cv.waitKey(0)  

# THRESHHOLDING

#simple thresholding

threshold , thresh = cv.threshold(gray,150,255,cv.THRESH_BINARY)
#cv.imshow("simple thresholded",thresh)

threshold , thresh_inv = cv.threshold(gray, 150 , 255 , cv.THRESH_BINARY_INV)
#cv.imshow("simple thresholded INV",thresh_inv)
#cv.waitKey(0)

# Adaptive Thresholding

# Mean Adaptive Threshold
adaptive_mean = cv.adaptiveThreshold(
    gray,
    maxValue=255,
    adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
    thresholdType=cv.THRESH_BINARY,
    blockSize=11,   # size of neighbourhood
    C=2             # constant subtracted
)

# Gaussian Adaptive Threshold
adaptive_gauss = cv.adaptiveThreshold(gray,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

# cv.imshow("Adaptive Mean", adaptive_mean)
# cv.imshow("Adaptive Gaussian", adaptive_gauss)
# cv.waitKey(0)

## Use Gaussian when lighting changes smoothly

## blockSize must be odd

# Edge Detection

 # Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
# cv.imshow("Laplacian", lap)
 
 #Sobel Gradient
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1)

# cv.imshow("Sobel X", sobel_x)
# cv.imshow("Sobel Y", sobel_y) 

combined_sobel = cv.bitwise_or(sobel_x,sobel_y)
 
 #canny
edges = cv.Canny(img, 100, 200)
# cv.imshow("Edges", edges)
# cv.waitKey(0)

# CONTOUR DETECTION
# Threshold first
thresholdd, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)

# or use canny , blur 
# Find contours
contours, hierarchy = cv.findContours(
    thresh,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

print(f"Total contours found: {len(contours)}")

# Draw contours
contour_img = img.copy()
cv.drawContours(contour_img, contours, -1, (0,255,0), 2)

#cv.imshow("Contours", contour_img)
#cv.waitKey(0)

#Saving image

# cv.imwrite('name.png',img)
