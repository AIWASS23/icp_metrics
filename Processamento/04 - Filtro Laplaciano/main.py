import cv2

src = cv2.imread('Borboleta.jpg', 0)
out = cv2.imread('Borboleta.jpg', 0)
media = cv2.imread('Borboleta.jpg', 0)

def laplaciano(kernel):
    for x in range(1, 271):
        for y in range(1, 334):
            media[x, y] = (src[x-1, y-1]/9 + src[x, y-1]/9 + src[x+1, y-1]/9 + 
                            src[x, y]/9 + src[x-1, y]/9 + src[x+1, y]/9 + 
                            src[x-1, y+1]/9 + src[x, y+1]/9 + src[x+1, y+1]/9)

    if kernel == -4:
        for x in range(1, 271):
            for y in range(1, 334):
                out[x, y] = (media[x, y] * -4 + media[x, y-1] + media[x-1, y] + media[x+1, y] + media[x, y+1])  
                out[x, y] += 100
                                
    elif kernel == 4:
        for x in range(1, 271):
            for y in range(1, 334):
                out[x, y] = (media[x, y] * 4 - media[x, y-1] - media[x-1, y] - media[x+1, y] - media[x, y+1])  
                out[x, y] += 100
                        
                        
    elif kernel == -8:
        for x in range(1, 271):
            for y in range(1, 334):
                out[x, y] = (media[x, y] * -8 + media[x, y-1] + media[x-1, y] +  media[x+1, y] + media[x, y+1] + 
                             media[x-1, y-1] + media[x+1, y-1] + media[x-1, y+1] + media[x+1, y+1]) 
                out[x, y] += 100                     
                        
    elif kernel == 8:
        for x in range(1, 271):
            for y in range(1, 334):
                out[x, y] = (media[x, y] * 8 - media[x, y-1] - media[x-1, y] - media[x+1, y] - media[x, y+1] - 
                             media[x-1, y-1] - media[x+1, y-1] - media[x-1, y+1] - media[x+1, y+1]) 
                out[x, y] += 100
            
    return cv2.imshow('output', out)

laplaciano(-4)
cv2.waitKey(0)
cv2.destroyAllWindows()