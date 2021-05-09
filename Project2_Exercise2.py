import cv2
import random
import math
from matplotlib import pyplot as plt

"""
------Question 1-----------
"""
# import the image from the image folder
img = cv2.imread("images/trust.jpg")

# apply the gaussian blur
blur = cv2.GaussianBlur(img, (11, 11), 0)
widht, height = img.shape[:2]

"""k = 1000

for x in range(widht):
    for y in range(height):
        val = math.exp(-k*(x**2 + y**2)**(5/6))
        print(val)
"""

"""
------Question 2-----------
"""

"""
function that adds random noise to and image
Since we want salt and pepper noise,
we chose random pixel in the image and color them white,
same goes for black pixels. 
"""


def random_noise(img):
    # get the dimensions of the inputted image
    w, h = img.shape[:2]

    # pick a random number of pixel to change
    nb_of_pixels = random.randint(150000, 200000)

    for i in range(nb_of_pixels):
        # random x and y coordinates
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)

        # color that pixel in white
        img[y][x] = 255

    # apply the same technique for black pixels
    for i in range(nb_of_pixels):
        # random x and y coordinates
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)

        # color that pixel in black
        img[y][x] = 0

    return img

# apply noise to our image
random_noise(blur)

# display all images
cv2.imshow("initial image", img)
cv2.imshow("blurred image", blur)
cv2.waitKey(0)
