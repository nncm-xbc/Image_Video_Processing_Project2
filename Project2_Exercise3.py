import cv2
import numpy as np
from matplotlib import pyplot as plt

# import image
image = cv2.imread("images/flower.jpeg", 0)

# read dimensions of the image
h, w = image.shape[:2]

# create translation matrix with a translation of 200 pixels to the right
translation_matrix = np.float32([[1, 0, 600], [0, 1, 0]])

# apply the translation matrix to the image
translated_image = cv2.warpAffine(image, translation_matrix, (w, h))

# 2D FFT of the original image
fft1 = np.fft.fft2(image)

# 2D FFT of the translated image
fft2 = np.fft.fft2(translated_image)

shiftfft1 = np.fft.fftshift(fft1)
shiftfft2 = np.fft.fftshift(fft2)

plt.subplot(141), plt.imshow(image, cmap='gray')
plt.title('original image '), plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(translated_image, cmap='gray')
plt.title('translated image'), plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(np.log(1+np.abs(shiftfft1)), cmap='gray')
plt.title('ft image'), plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(np.log(1+np.abs(shiftfft2)), cmap='gray')
plt.title('translated ft image'), plt.xticks([]), plt.yticks([])
plt.show()


rows, cols = fft1.shape[:2]
fig1, x = plt.subplots(nrows=1, ncols=1)

nVals = np.arange(start = -rows/2, stop = rows/2)* 300/rows
x.plot(nVals, np.abs(np.fft.fftshift(fft1[:, 1])))

x.set_title('Double Sided FFT')
x.set_xlabel('Sample points (N-point DFT)')
x.set_ylabel('DFT Values')
x.set_xlim(-50, 50)
x.set_xticks(np.arange(-50, 50+10, 10))
fig1.show()



rows, cols = fft2.shape[:2]
fig2, x = plt.subplots(nrows=1, ncols=1)

nVals = np.arange(start = -rows/2, stop = rows/2)* 300/rows
x.plot(nVals, np.abs(np.fft.fftshift(fft2[:, 1])))

x.set_title('Double Sided FFT')
x.set_xlabel('Sample points (N-point DFT)')
x.set_ylabel('DFT Values')
x.set_xlim(-50, 50)
x.set_xticks(np.arange(-50, 50+10, 10))
fig2.show()
