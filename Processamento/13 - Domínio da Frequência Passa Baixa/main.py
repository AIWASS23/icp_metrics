import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('bear.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows, cols = img.shape
crow,ccol = rows/2 , cols/2

mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

x = cv2.getGaussianKernel(5,10)
gaussian = x*x.T

fft_filters = np.fft.fft2(gaussian) 
fft_shift = np.fft.fftshift(fft_filters)  
mag_spectrum = np.log(np.abs(fft_shift)+1)

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Source'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Espectro da Imagem'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img_back, cmap = 'gray')
plt.title('Resultado Final'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(mag_spectrum,cmap = 'gray')
plt.title('Espectro do Filtro Gaussiano'), plt.xticks([]), plt.yticks([])
plt.show()