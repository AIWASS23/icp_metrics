import cv2 

src = cv2.imread('Limiar.jpg', 0)

def limiarBin():
    T = 150
    for x in range(1, 374):
        for y in range(1, 499):
            if src[x,y] > T:
                src[x,y] = 0
            
            elif src[x,y] <= T:
                src[x,y] = 255
                
    cv2.imwrite('out.jpg', src)
    return cv2.imshow('output', src)

limiarBin()
cv2.waitKey(0)
cv2.destroyAllWindows()