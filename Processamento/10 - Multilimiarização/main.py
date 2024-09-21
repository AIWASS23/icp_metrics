import cv2 

src = cv2.imread('Limiar.jpg', 0)

def limiarBin():
    T1 = 150
    T2 = 200
    for x in range(1, 374):
        for y in range(1, 499):
            if src[x,y] < T1:
                src[x,y] = 255
            
            elif src[x,y] < T2:
                src[x,y] = 128
            
            elif src[x,y] >= T2:
                src[x,y] = 0
    
    cv2.imwrite('out.jpg', src)
    return cv2.imshow('output', src)

limiarBin()
cv2.waitKey(0)
cv2.destroyAllWindows()