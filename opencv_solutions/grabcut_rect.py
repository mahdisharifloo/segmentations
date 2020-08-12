import numpy as np 
import cv2

img_path = input('input image path : ')
img = cv2.imread(img_path)
img = cv2.resize(img,(224,224))
    
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)

fgdModel = np.zeros((1,65),np.float64)

height, width = img.shape[:2]

rect = (50,10,width-100,height-20)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img2 = img*mask2[:,:,np.newaxis]

img2[mask2 == 0] = (255, 255, 255)
    
final = np.ones(img.shape,np.uint8)*0 + img2
    
# cv2.imwrite('img2.jpg', img2)
# cv2.imwrite('final.jpg', final)
cv2.imwrite('final.jpg', final)
