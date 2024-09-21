import cv2

src= cv2.imread('Castle.png', 0)
out = cv2.imread('Castle.png', 0)

for x in range(1, 480):
    for y in range(1, 320):
        out[x, y] = (src[x-1, y-1]/16 + src[x, y-1]/16 * 2 + src[x+1, y-1]/16 + 
                     src[x-1, y]/16 * 2 + src[x, y]/16 * 4 + src[x+1, y]/16 * 2 + 
                     src[x-1, y+1]/16 + src[x, y+1]/16 * 2 + src[x+1, y+1]/16)
     
cv2.imshow('output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()