import cv2

src = cv2.imread('Bike.jpg', 0)
sh = cv2.imread('Bike.jpg', 0)
sv = cv2.imread('Bike.jpg', 0)
out = cv2.imread('Bike.jpg', 0)

for x in range(1, 479):
    for y in range(1, 639):
        sh[x, y] = (src[x-1, y-1]/4 - src[x+1, y-1]/4 
                    + 2*src[x-1, y]/4 - 2*src[x+1, y]/4 
                    +src[x-1, y+1]/4 - src[x+1, y+1]/4)
                
        sv[x, y] = (-src[x-1, y-1]/4 - 2*src[x, y-1]/4 - src[x+1, y-1]/4  
                     +src[x-1, y+1]/4 + 2*src[x, y+1]/4 + src[x+1, y+1]/4)
    
        out[x,y] = sh[x,y] + sv[x,y]
    
cv2.imshow('output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()