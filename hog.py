import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt 
import imutils
import math
from scipy import ndimage

def foreground(image):
    
    boundaries = [([0,0,0],[100,255,100])]
# loop over the boundaries
    for (lower, upper) in boundaries:
# create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
 
# find the colors within the specified boundaries and apply
# the mask
        mask = cv2.inRange(image, lower,upper)
        output = cv2.bitwise_and(image, image, mask = mask)
    

    return output

# construct the argument parse and parse the arguments


MIN_AREA=200
test=5

hog = cv.HOGDescriptor()
    
image_path = "test/" + "23" + ".jpg"
# load the image
image = cv.imread(image_path)
fixed_size = tuple((500, 500))

image = cv.resize(image, fixed_size)
image = foreground(image)

#boundaries = [([0,0,0],[100,255,100])]
#grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#h = hog.compute(grey)

cnt=image

leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

cv.imshow("hog", cnt)
cv.waitKey(0)
cv.destroyAllWindows()
'''
print(h)
print(len(h))
'''