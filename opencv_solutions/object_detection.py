# Imports
import numpy as np
import cv2
import time

# Read Image & Convert
img = cv2.imread('/home/mahdi/7.jpg')
result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Filtering
lower = np.array([1,60,50])
upper = np.array([255,255,255])
result = cv2.inRange(result, lower, upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
result = cv2.dilate(result,kernel)

# Contours
contours, hierarchy = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
if len(contours) != 0:
    for (i, c) in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 100000:
            print(area)
            cv2.drawContours(img, c, -1, (255,255,0), 12)
            x,y,w,h = cv2.boundingRect(c)            
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),12)

# Stack results
result = np.vstack((result, img))
resultOrig = result.copy()

# Save image to file before resizing
cv2.imwrite(str(time.time())+'_0_result.jpg',resultOrig)

# Resize
max_dimension = float(max(result.shape))
scale = 900/max_dimension
result = cv2.resize(result, None, fx=scale, fy=scale)

# Show results
cv2.imshow('res',result)

cv2.waitKey(0)
cv2.destroyAllWindows()
