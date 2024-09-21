import cv2
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('T.jpg', 0)

trans = cv2.warpAffine(src, np.float32([[1,0,100],[0,1,50]]), (288, 288))
rot = dst = cv2.warpAffine(src, cv2.getRotationMatrix2D((144, 144), 90, 1),(288,288))
esc = cv2.resize(src, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

plt.subplot(131), plt.imshow(trans), plt.title('Translacao')
plt.subplot(132), plt.imshow(rot), plt.title('Rotacao')
plt.subplot(133), plt.imshow(esc), plt.title('Escala')

plt.show()