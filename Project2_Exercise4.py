import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# import image
img = cv2.imread('images/flower.jpeg', 0)

# fourier transform (frequency domain) and shift
ftimg = np.fft.fft2(img)
centered = np.fft.fftshift(ftimg)
centered_plot = np.log(1+np.abs(centered))

# function that inputs an image and outputs the corresponding filter
def Filter(img):

    rows, cols = img.shape
    mask = img.copy()

    j, x, y = 0.001, 10, 10

    # for all frequency values compute the value of the filter
    for r in range(rows):
        for c in range(cols):

            h = math.exp(-(j*r*x))*math.exp(-(j*c*y))
            mask[r, c] = h

    # return the final mask
    return mask


# call the filter on the centered frequency image and apply it to the frequency domain image.
uncentered_mask = Filter(centered)
mask = np.fft.fftshift(uncentered_mask)
proceced_img_ft = mask * centered

final_img = np.fft.ifft2(np.fft.ifftshift(proceced_img_ft))


plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('initial image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(np.log(np.abs(ftimg)), cmap='gray')
plt.title('ft image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(centered_plot, cmap='gray')
plt.title('centered ft image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(141), plt.imshow(np.log(1+np.abs(uncentered_mask)), cmap='gray')
plt.title('uncentered mask'), plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(np.log(1+np.abs(mask)), cmap='gray')
plt.title('mask'), plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(np.log(1+np.abs(proceced_img_ft)), cmap='gray')
plt.title('mask + image'), plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(np.abs(final_img), cmap='gray')
plt.title('final image'), plt.xticks([]), plt.yticks([])
plt.show()
