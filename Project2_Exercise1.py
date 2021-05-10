import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image

# original images in greyscale
original1grey = cv2.imread("images/kitten.jpg", 0)
original2grey = cv2.imread("images/puppy.jpg", 0)

# apply the fourier transform to both images
fourier1 = np.fft.fft2(original1grey)
fourier2 = np.fft.fft2(original2grey)

# center both images
shift1 = np.fft.fftshift(fourier1)
shift2 = np.fft.fftshift(fourier2)

# improve the quality of the image for the ploting
frequencyImg1 = np.log(np.abs(shift1))
frequencyImg2 = np.log(np.abs(shift2))

# function that applies a low-pass filter to a given image
def low_pass_filter(img):
    rows, cols = img.shape

    appliedlow = img

    # compute the center of the image
    center_y = rows/2
    center_x = cols/2

    # for all pixels in the image compute the distance to the center of the image
    for i in range(rows):
        for j in range(cols):

            distance = math.sqrt(((i-center_y)**2) + ((j-center_x)**2))

            # if the distance exceeds a threshold change the value of the pixel to 0 (black)
            if distance >= 60:
                appliedlow[i, j] = 0

    # return the modified image
    return appliedlow

# function that applies a high-pass filter to a given image
def high_pass_filter(img):
    rows, cols = img.shape

    appliedhigh = img

    # compute the center of the image
    center_y = rows/2
    center_x = cols/2

    # for all pixels in the image compute the distance to the center of the image
    for i in range(rows):
        for j in range(cols):

            distance = math.sqrt(((i-center_y)**2) + ((j-center_x)**2))

            # if the distance does not meet a threshold change the value of the pixel to 0 (black)
            if distance <= 60:
                appliedhigh[i, j] = 0
    # return the modified image
    return appliedhigh

# create the low and high pass filters
lowpassfilt = low_pass_filter(shift1)
highpassfilt = high_pass_filter(shift2)

# apply our filters to to the images in the frequency domain
lowpass_img = lowpassfilt * shift1
highpass_img = highpassfilt * shift2

# addition in the frequency domain
frequencyadd = lowpassfilt + highpassfilt
inverseshift = np.fft.ifftshift(frequencyadd)
inversefourier = np.fft.ifft2(inverseshift)

# addition in the spatial domain
spatiallow = np.fft.ifft2(np.fft.fftshift(lowpassfilt))
spatialhigh = np.fft.ifft2(np.fft.fftshift(highpassfilt))
spatialadd = spatialhigh + spatiallow


# Mean Squared Error
def mse(img1, img2):
    return np.sqrt(((img1 - img2) ** 2).mean())

error_val = mse(inversefourier, spatialadd)
print("Mean Squared Error :", error_val)

"""-------plot1----------"""
plt.subplot(121), plt.imshow(original1grey, cmap='gray')
plt.title('Kitten image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(frequencyImg1, cmap='gray')
plt.title('Frequency image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121), plt.imshow(original2grey, cmap='gray')
plt.title('Puppy image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(frequencyImg2, cmap='gray')
plt.title('Frequency image'), plt.xticks([]), plt.yticks([])
plt.show()
"""-------plot1----------"""

"""-------plot2----------"""
plt.subplot(121), plt.imshow(np.log(1+np.abs(lowpass_img)), cmap='gray')
plt.title('low-pass'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.log(1+np.abs(highpass_img)), cmap='gray')
plt.title('high-pass'), plt.xticks([]), plt.yticks([])
plt.show()
"""-------plot2----------"""

"""-------plot3----------"""
plt.subplot(121), plt.imshow(np.log(1+np.abs(inversefourier)), cmap='gray')
plt.title('frequency addition'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.log(1+np.abs(spatialadd)), cmap='gray')
plt.title('spatial addition'), plt.xticks([]), plt.yticks([])
plt.show()
"""-------plot3----------"""


"""-------plot4----------"""
shifted_fft2 = np.fft.fftshift(fourier1)
rows, cols = shifted_fft2.shape[:2]
fig1, x = plt.subplots(nrows=1, ncols=1)

nVals = np.arange(start = -rows/2, stop = rows/2)* 300/rows
x.plot(nVals, np.abs(shifted_fft2[:, 1]))

x.set_title('Double Sided FFT')
x.set_xlabel('Sample points (N-point DFT)')
x.set_ylabel('DFT Values')
x.set_xlim(-50, 50)
x.set_xticks(np.arange(-50, 50+10, 10))
fig1.show()


shifted_fft2 = np.fft.fftshift(fourier2)
rows, cols = shifted_fft2.shape[:2]
fig2, x = plt.subplots(nrows=1, ncols=1)

nVals = np.arange(start = -rows/2, stop = rows/2)* 300/rows
x.plot(nVals, np.abs(shifted_fft2[:, 1]))

x.set_title('Double Sided FFT')
x.set_xlabel('Sample points (N-point DFT)')
x.set_ylabel('DFT Values')
x.set_xlim(-50, 50)
x.set_xticks(np.arange(-50, 50+10, 10))
fig2.show()


shifted_fft2 = np.fft.fftshift(frequencyadd)
rows, cols = shifted_fft2.shape[:2]
fig3, x = plt.subplots(nrows=1, ncols=1)

nVals = np.arange(start = -rows/2, stop = rows/2)* 300/rows
x.plot(nVals, np.abs(shifted_fft2[:, 1]))

x.set_title('Double Sided FFT')
x.set_xlabel('Sample points (N-point DFT)')
x.set_ylabel('DFT Values')
x.set_xlim(-50, 50)
x.set_xticks(np.arange(-50, 50+10, 10))
fig3.show()
"""-------plot4----------"""
