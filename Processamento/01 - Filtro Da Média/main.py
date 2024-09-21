import cv2

src = cv2.imread('Castle.png', 0)
out = cv2.imread('Castle.png', 0)

def media(kernel): 
    if kernel == 3:
        for x in range(1, 480):
            for y in range(1, 320):
                out[x, y] = (src[x-1, y-1]/9 + src[x, y-1]/9 + src[x+1, y-1]/9 + 
                             src[x-1, y]/9 + src[x, y]/9 + src[x+1, y]/9 + 
                             src[x-1, y+1]/9 + src[x, y+1]/9 + src[x+1, y+1]/9)
        
    elif kernel == 5:
        for x in range(1, 479):
            for y in range(1, 319):
                out[x, y] = (src[x-1, y-1]/25 + src[x, y-1]/25 + src[x+1, y-1]/25 + src[x, y]/25 + src[x-1, y]/25 + 
                             src[x+1, y]/25 + src[x-1, y+1]/25 + src[x, y+1]/25 + src[x+1, y+1]/25 + src[x-2, y-2]/25 + 
                             src[x-1, y-2]/25 + src[x, y-2]/25 + src[x+1, y-2]/25 + src[x+2, y-2]/25 +src[x-2, y-1]/25 + 
                             src[x+2, y-1]/25 + src[x-2, y]/25 + src[x+2, y]/25 +src[x-2, y+1]/25 + src[x+2, y+1]/9 + 
                             src[x-2, y+2]/25 + src[x-1, y+2]/25 + src[x, y+2]/25 + src[x+1, y+2]/25 + src[x+2, y+2]/25) 
        
    elif kernel == 7:
        for x in range(1, 478):
            for y in range(1, 318):
                out[x, y] = ((src[x-1, y-1]/49) + (src[x, y-1]/49) + (src[x+1, y-1]/49) + (src[x, y]/49) + (src[x-1, y]/49) + (src[x+1, y]/49) + (src[x-1, y+1]/49) + 
                             (src[x, y+1]/49) + (src[x+1, y+1]/49) + (src[x-2, y-2]/49) + (src[x-1, y-2]/49) + (src[x, y-2]/49) + (src[x+1, y-2]/49) + (src[x+2, y-2]/49) +
                             (src[x-2, y-1]/49) + (src[x+2, y-1]/49) + (src[x-2, y]/49) + (src[x+2, y]/49) +(src[x-2, y+1]/49) + (src[x+2, y+1]/49) + (src[x-2, y+2]/49) + 
                             (src[x-1, y+2]/49) + (src[x, y+2]/49) + (src[x+1, y+2]/49) + (src[x+2, y+2]/49) + (src[x-3, y-3]/49) + (src[x-2, y-3]/49) + (src[x-1, y-3]/49) + 
                             (src[x, y-3]/49) + (src[x+1, y-3]/49) + (src[x+2, y-3]/49) + (src[x+3, y-3]/49) + (src[x-3, y-2]/49) + (src[x+3, y-2]/49) + (src[x-3, y-1]/49) + 
                             (src[x+3, y-1]/49) + (src[x-3, y]/49) + (src[x+3, y]/49) + (src[x-3, y+1]/49) + (src[x+3, y+1]/49) + (src[x-3, y+2]/49) + (src[x+3, y+2]/49) + 
                             (src[x-3, y+3]/49) + (src[x-2, y+3]/49) + (src[x-1, y+3]/49) + (src[x, y+3]/49) + (src[x+1, y+3]/49) + (src[x+2, y+3]/49) + (src[x+3, y+3]/49))
        
    return cv2.imshow('output', out)

media(3)
cv2.waitKey(0)
cv2.destroyAllWindows()