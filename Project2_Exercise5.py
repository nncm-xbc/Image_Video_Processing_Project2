import cv2
import numpy as np
from matplotlib import pyplot as plt

# import images
imageA = cv2.imread("images/tile.jpg", 0)
imageB = cv2.imread("images/tile0.jpg", 0)

# Compute DFT of the two images
dftA = np.fft.fft2(imageA)
dftB = np.fft.fft2(imageB)

plt.subplot(121), plt.imshow(imageA, cmap='gray')
plt.title('Image A'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(imageB, cmap='gray')
plt.title('Image B'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121), plt.imshow(np.log(1+np.abs(dftA)), cmap='gray')
plt.title('Spectral appearance of A'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.log(1+np.abs(dftB)), cmap='gray')
plt.title('Spectral appearance of B'), plt.xticks([]), plt.yticks([])
plt.show()
