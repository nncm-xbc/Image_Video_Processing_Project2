import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import random
"""
------Question 1-----------
"""
# import the image from the image folder
img = cv2.imread("images/trust.jpg", 0)

# apply the gaussian blur
# blur = cv2.GaussianBlur(img, (11, 11), 0)
# widht, height = img.shape[:2]

# frenquency domain of the initial image
fft = np.fft.fft2(img)
shifted_fft = np.fft.fftshift(fft)

# function that computes a gaussian blur mask given an image
def gaussian_filter(img):
    k = 0.00002
    rows, cols = img.shape

    mask = img.copy()

    for x in range(cols):
        for y in range(rows):
            val = math.exp(-k*(x**2 + y**2)**(5/6))
            mask[y, x] = val

    return mask


# apply the mask to the frequency domain image
mask = np.fft.fftshift(gaussian_filter(shifted_fft))
blurred_img = shifted_fft * mask

# compute back to spatial domain for ploting
proceced_img = np.fft.ifft2(np.fft.ifftshift(blurred_img))


"""
function that adds random noise to and image
Since we want salt and pepper noise,
we chose random pixel in the image and color them white,
same goes for black pixels. 
"""
def salt_and_pepper(img):
    # get the dimensions of the inputted image
    w, h = img.shape[:2]

    seasoned_img = img.copy()

    # pick a random number of pixel to change
    nb_of_pixels = random.randint(150000, 200000)

    for i in range(nb_of_pixels):
        # random x and y coordinates
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)

        # color that pixel in white
        seasoned_img[y][x] = 255

    # apply the same technique for black pixels
    for i in range(nb_of_pixels):
        # random x and y coordinates
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)

        # color that pixel in black
        seasoned_img[y][x] = 0

    return seasoned_img


# apply noise to our image
seasoned_img = salt_and_pepper(proceced_img)


plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('original image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(np.abs(proceced_img), cmap='gray')
plt.title('processed image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(np.abs(seasoned_img), cmap='gray')
plt.title('seasoned image'), plt.xticks([]), plt.yticks([])
plt.show()
