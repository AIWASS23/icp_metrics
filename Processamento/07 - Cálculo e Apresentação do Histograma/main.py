import cv2
import matplotlib.pyplot as plt
import numpy as np

src = cv2.imread('Table.jpg', 0)

eixo_y = [0] * 256

for x in range(1, 374):
    for y in range(1, 499):  
        a = src[x,y] 
        eixo_y[src[x,y]] = eixo_y[src[x,y]] + 1

plt.subplot(111).bar(np.arange(256), eixo_y, color = 'black')
plt.show()