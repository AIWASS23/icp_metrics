import cv2
import numpy as np

src = cv2.imread('bear.jpg', 0) #gray image
srcHist = cv2.calcHist([src], [0], None, [256], [0, 256])

def correcao_gama(src, correcao):
    src = src/255.0
    src = cv2.pow(src, correcao)
    return np.uint8(src*255)

out = correcao_gama(src, 0.7)

cv2.imshow('out', out)
cv2.waitKey(0)