import cv2
import numpy as np
from matplotlib import pyplot as plt

# import image
image = cv2.imread("images/flower.jpeg", 0)

# read dimensions of the image
h, w = image.shape[:2]

# create translation matrix with a translation of 200 pixels to the right
translation_matrix = np.float32([[1, 0, 200], [0, 1, 0]])

# apply the translation matrix to the image
translated_image = cv2.warpAffine(image, translation_matrix, (w, h))

# 2D FFT of the translated image
fft = np.fft.fft2(translated_image)

plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('original image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(translated_image, cmap='gray')
plt.title('translated image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.plot(fft[:, 0], fft[:, 1], "b")
plt.title('magnitude 2d fft'), plt.xticks([]), plt.yticks([])
plt.show()
