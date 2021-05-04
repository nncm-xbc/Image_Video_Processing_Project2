import cv2
from matplotlib import pyplot as plt

#import the image from the image folder
img = cv2.imread("images/trust.jpg")

#apply the gaussian blur
blur = cv2.GaussianBlur(img,(11,11),0)

cv2.imshow("initial image",img)
cv2.imshow("blurred image",blur)
cv2.waitKey(0)
