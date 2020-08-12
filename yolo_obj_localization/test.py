import cv2 
import numpy as np 
import matplotlib.pyplot as plt

image = cv2.imread('/home/mahdi/projects/dgkala/logo_detection/darknet/predictions.jpg')

imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

ret , thresh = cv2.threshold(imgray,127,255,0)

contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

imconture = cv2.drawContours(image,contours,3,(0,0,255),3)

cv2.imwrite('out.jpg',imconture)


def mask_generator_white(img):
    '''
    this function find the white aria and remove them and create mask with that
    '''
    white_lower = np.asarray([230, 230, 230])
    white_upper = np.asarray([255, 255, 255])

    mask = cv2.inRange(img, white_lower, white_upper)
    mask = cv2.bitwise_not(mask)

    cv2.imwrite('mask.jpg',mask)
    return mask
def find_bigest_rect_white(img,mask):
    '''
    this function find contours in this white mask and select the largest one.
    '''
    cnt, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(cnt, key=lambda x:cv2.contourArea(x))
    bounding_rect = cv2.boundingRect(largest_contour)

    cropped_image = img[bounding_rect[1]: bounding_rect[1]+bounding_rect[3],
                    bounding_rect[0]:bounding_rect[0]+bounding_rect[2]]
    return cropped_image