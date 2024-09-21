import cv2 
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('j.png', 0)

kernel = np.matrix([[0, 0, 1, 0, 0],     #Cruz
                    [0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0]], np.uint8)

gradiente = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, kernel)

plt.subplot(121), plt.imshow(src, cmap = 'gray'), plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(gradiente, cmap = 'gray'), plt.title('Gradiente Morfologico'), plt.xticks([]), plt.yticks([])

plt.show() 