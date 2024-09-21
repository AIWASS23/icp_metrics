import cv2
from matplotlib import pyplot as plt

src = cv2.imread('bear2.jpg', 0)
srcHist = cv2.calcHist([src], [0], None, [256], [0,256])

out = cv2.imread('bear2.jpg', 0)

b = 20  #Controle de Brilho 

for x in range(1, 240):
    for y in range(1, 300): 
        out[x,y] = src[x,y] - b

outHist = cv2.calcHist([out], [0], None, [256], [0,256])