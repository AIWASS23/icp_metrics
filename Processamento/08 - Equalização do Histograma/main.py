import cv2
import matplotlib.pyplot as plt
import numpy as np

src = cv2.imread('Table.jpg', 0)

Pr = [0] * 256
PrF = [0] * 256
PrAc = 0

for x in range(1, 374): #Calculo do Histograma
    for y in range(1, 499):  
        Pr[src[x,y]] = Pr[src[x,y]] + 1

for i in range(0, 256): #Calculo de probabilidade e probabilidade acumulada
    PrAc = PrAc + Pr[i] / 187500.0
    PrF[i] = round(PrAc * 255, 0)
    print PrF[i]

for x in range(1, 374):
    for y in range(1, 499): 
        src[x,y] = PrF[src[x,y]]

plt.subplot(111).bar(np.arange(256), PrF, color = 'black')
plt.show()

cv2.imshow('output', src)
cv2.waitKey(0)
cv2.destroyAllWindows()