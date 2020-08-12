# -*- coding: utf-8 -*-

import cv2
import numpy as np
#import rectangle base on the input selection 
def draw_rectangle(event,x,y,flags,params):
    global x_init,y_init,drawing,top_left_pt,bottom_right_pt,img_orig
    #detecting mouse button down event
    if event == cv2.EVENT_LBUTTONDOWN:
          drawing = True
          x_init,y_init = x,y
    #detecting mouse movement 
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            top_left_pt,bottom_right_pt = (x_init,y_init),(x,y)
            img[y_init:y,x_init:x] = 255- img_orig[y_init:y,x_init:x]
            cv2.rectangle(img,top_left_pt,bottom_right_pt,(0,255,0),2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        top_left_pt,bottom_right_pt =  (x_init,y_init),(x,y)
        img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]
        cv2.rectangle(img, top_left_pt, bottom_right_pt, (0,255,0), 2)
        rect_final = (x_init, y_init, x-x_init, y-y_init)
        # Run Grabcut on the region of interest
        run_grabcut(img_orig, rect_final)
        


# Grabcut algorithm
def run_grabcut(img_orig, rect_final):
    #initialize the mask 
    mask = np.zeros(img_orig.shape[:2],np.uint8)
    #extract the rectangle and set the region of interest in the above mask 
    x,y,w,h = rect_final
    mask[x:y+h,x:x+w] = 1
    # Initialize background and foreground models
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    # Run Grabcut algorithm
    cv2.grabCut(img_orig,mask,rect_final,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    #extract new mask 
    mask2 = np.where((mask==2)|(mask == 0),0,1).astype('uint8')
    #apply the above mast to the image 
    img_orig = img_orig*mask2[:,:,np.newaxis]
    #display the image 
    cv2.imshow('output',img_orig)
    
if __name__ == '__main__':
    drawing = False
    top_left_pt,bottom_right_pt = (-1,-1),(-1,-1)
    #read the input image 
    img_orig = cv2.imread(input('image path : '))
    img = img_orig.copy()
    cv2.namedWindow('input')
    cv2.setMouseCallback('input',draw_rectangle)
    
    while True:
        cv2.imshow('input',img)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cv2.destroyAllWindows()
        
