import cv2
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('j.png',0)

elementoEstruturante1 = np.matrix([[1, 1, 1, 1, 1],     #Retangulo
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1]], np.uint8)

elementoEstruturante2 = np.matrix([[0, 1, 1, 1, 0],     #Circulo
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [0, 1, 1, 1, 0]], np.uint8)

elementoEstruturante3 = np.matrix([[0, 0, 1, 0, 0],     #Cruz
                                   [0, 0, 1, 0, 0],
                                   [1, 1, 1, 1, 1],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 1, 0, 0]], np.uint8)

erosion1 = cv2.erode(src, elementoEstruturante1)
erosion2 = cv2.erode(src, elementoEstruturante2)
erosion3 = cv2.erode(src, elementoEstruturante3)

plt.subplot(221), plt.imshow(src, cmap = 'gray'), plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(erosion1, cmap = 'gray'), plt.title('Flat'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(erosion2, cmap = 'gray'), plt.title('Circle'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(erosion3, cmap = 'gray'), plt.title('Plus'), plt.xticks([]), plt.yticks([])

plt.show()
##################################################################################################################################################################

dilation1 = cv2.dilate(src, elementoEstruturante1)
dilation2 = cv2.dilate(src, elementoEstruturante2)
dilation3 = cv2.dilate(src, elementoEstruturante3)

plt.subplot(221), plt.imshow(src, cmap = 'gray'), plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(dilation1, cmap = 'gray'), plt.title('Flat'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(dilation2, cmap = 'gray'), plt.title('Clircle'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(dilation3, cmap = 'gray'), plt.title('Plus'), plt.xticks([]), plt.yticks([])

plt.show()