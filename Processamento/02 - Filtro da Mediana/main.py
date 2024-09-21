import cv2

src= cv2.imread('Castle.png', 0)
out = cv2.imread('Castle.png', 0)

def mediana(kernel):
    if kernel == 3:
        for x in range(1, 480):
            for y in range(1, 320):
                l = [src[x-1, y-1], src[x, y-1], src[x+1, y-1], 
                     src[x, y], src[x-1, y], src[x+1, y], 
                     src[x-1, y+1], src[x, y+1], src[x+1, y+1]]
                l.sort()
                out[x, y] = l[4]
        
    elif kernel == 5:
        for x in range(1, 478):
            for y in range(1, 319):
                l = [src[x-1, y-1], src[x, y-1], src[x+1, y-1], src[x-1, y],src[x, y], 
                     src[x+1, y], src[x-1, y+1], src[x, y+1], src[x+1, y+1], src[x-2, y-2], 
                     src[x-1, y-2],  src[x, y-2], src[x+1, y-2], src[x+2, y-2], src[x-2, y-1], 
                     src[x+2, y-1], src[x-2, y], src[x+2, y], src[x-2, y+1], src[x+2, y+1],
                     src[x-2, y+2], src[x-1, y+2], src[x, y+2], src[x+1, y+2], src[x+2, y+2]]
                l.sort()
                out[x, y] = l[12]
                    
    elif kernel == 7:
        for x in range(1, 478):
            for y in range(1, 318):
                l = [src[x-1, y-1], src[x, y-1], src[x+1, y-1], src[x-1, y],src[x, y], src[x+1, y], src[x-1, y+1], 
                     src[x, y+1], src[x+1, y+1], src[x-2, y-2], src[x-1, y-2],  src[x, y-2], src[x+1, y-2], src[x+2, y-2], 
                     src[x-2, y-1], src[x+2, y-1], src[x-2, y], src[x+2, y], src[x-2, y+1], src[x+2, y+1],src[x-2, y+2], 
                     src[x-1, y+2], src[x, y+2], src[x+1, y+2], src[x+2, y+2], src[x-3, y-3], src[x-2, y-3], src[x-1, y-3], 
                     src[x, y-3], src[x+1, y-3], src[x+2, y-3], src[x+3, y-3], src[x-3, y-2], src[x+3, y-2], src[x-3, y-1], 
                     src[x+3, y-1], src[x-3, y], src[x+3, y], src[x-3, y+1],src[x+3, y+1], src[x-3, y+2], src[x+3, y+2], 
                     src[x-3, y+3], src[x-2, y+3], src[x-1, y+3], src[x, y+3],src[x+1, y+3], src[x+2, y+3], src[x+3, y+3]]
                l.sort()
                out[x, y] = l[24]
                                 
    return cv2.imshow('output', out)

mediana(3)       
cv2.waitKey(0)
cv2.destroyAllWindows()