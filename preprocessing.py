# import the necessary packages
import numpy as np
import cv2



MIN_AREA=200


def foreground(image):
    
    boundaries = [([0,40,31],[100,255,100])]
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




def cropping(image,output):
    try:
        imgBlurred = cv2.GaussianBlur(output,(5,5),0)
        hsv = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2HSV)
        h, s, v= cv2.split(hsv)
        thresh = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,2)
        #_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)  '
        imgThreshCopy = thresh.copy()
        #cv2.imshow("aaa",imgThreshCopy)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        imgContours, npaContours, npaHierarchy= cv2.findContours(thresh,
                                                             cv2.RETR_EXTERNAL,
                                                             cv2.CHAIN_APPROX_SIMPLE)
        MAX=0
    
        for npaContour in npaContours:
            if cv2.contourArea(npaContour)> MIN_AREA and cv2.contourArea(npaContour)> MAX :
                [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
                cv2.rectangle(output,(intX, intY),(intX+intW, intY+intH),(0,0,255),2)
                #imgROI = imgThreshCopy[intY:intY+intH, intX:intX+intW]
                #imgROI = imgThreshCopy[intY:intY+intH, intX:intX+intW]
                MAX=cv2.contourArea(npaContour)
                #cv2.imshow("imgROI", imgROI)
                #cv2.imshow("Image", image)
                cv2.waitKey(0)
                
            
            #cv2.imshow("imgroi", imgROI) 
        image = image[intY:intY+intH, intX:intX+intW] 
        
        #cv2.imwrite("New",image)
        #cv2.imshow("Image1", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return image,intX,intY
    except:
        return image,0,0
def rotate(image):
    try:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
    
        lines = cv2.HoughLines(edges,1,np.pi/180,200)
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
    
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.line(image,(480,0),(480,540),(0,0,255),2)

    
        return image
    except:
        return image

