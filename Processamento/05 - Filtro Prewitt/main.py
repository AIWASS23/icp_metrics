import cv2

src = cv2.imread('Bike.jpg', 0)
ph = cv2.imread('Bike.jpg', 0)
pv = cv2.imread('Bike.jpg', 0)
out = cv2.imread('Bike.jpg', 0)

for x in range(1, 479):
    for y in range(1, 639):     
        ph[x, y] = (src[x-1, y-1]/3 - src[x+1, y-1]/3 
                    +src[x-1, y]/3 - src[x+1, y]/3 
                    +src[x-1, y+1]/3 - src[x+1, y+1]/3)
                
        pv[x, y] = (-src[x-1, y-1]/3 - src[x, y-1]/3 - src[x+1, y-1]/3  
                    +src[x-1, y+1]/3 + src[x, y+1]/3 + src[x+1, y+1]/3)
    
        out[x,y] = ph[x,y] + pv[x,y]
        
cv2.imshow('output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()