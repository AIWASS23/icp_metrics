import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bear.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

rows, cols = img.shape
crow,ccol = rows/2 , cols/2
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

fft_shift = np.fft.fftshift(np.fft.fft2(laplacian))
mag_spectrum = np.log(np.abs(fft_shift)+1)

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Source'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Espectro da Imagem'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img_back, cmap = 'gray')
plt.title('Resultado do Filtro'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(mag_spectrum, cmap = 'gray')
plt.title('Espectro do Filtro Laplaciano'), plt.xticks([]), plt.yticks([])

plt.show()