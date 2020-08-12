# -*- coding: utf-8 -*-
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import subprocess
import os 
import re 
from PIL import Image 
    


def yolo_runner(image_path):
    # root_dir = '/home/mahdi/projects/dgkala/logo_detection/darknet/'
    root_dir = input('Input root directory of darknet'
                    '\nlike this : /home/mahdi/projects/dgkala/logo_detection/darknet/'
                    '\n[INPUT] : ')
    os.chdir(root_dir)
    try:      
        output = subprocess.Popen(['./darknet','detector','test','cfg/coco.data',
                        'cfg/yolov3.cfg','yolov3.weights',image_path],
                        stdin=subprocess.PIPE, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        data_bytes = output.communicate(bytes(str(output.stdout.name), 'utf-8'))[0]
        out = data_bytes.decode('utf-8', 'ignore')
    except :
        print('[ERROR] check your darknet installation or configurtion.')
    return out


def box_and_label(str): 
    n = len(re.findall(r'Box', str))
    boxes = []
    labels = []
    percents = []
    for i in range(0,n):
        label_ = re.findall(r'[A-Z,a-z]*: [0-9][0-9]%', str)[i]
        label_ = re.findall(r'[A-Z,a-z]*',label_)[0]
        percent = re.findall(r'[0-9][0-9]%', str)[i]
        Left_str = re.findall(r'Left=[0-9]+', str)[i]
        Top_str = re.findall(r'Top=[0-9]+', str)[i]
        Right_str = re.findall(r'Right=[0-9]+', str)[i] 
        Bottom_str = re.findall(r'Bottom=[0-9]+', str)[i] 
        left = int( re.findall(r'[0-9]+', Left_str)[0] )
        right = int(re.findall(r'[0-9]+', Right_str)[0])
        top = int(re.findall(r'[0-9]+', Top_str)[0])
        bottom = int(re.findall(r'[0-9]+', Bottom_str)[0])
        box = [left+1,top+1,right+1,bottom+1]
        labels.append(label_)
        percents.append(percent)
        boxes.append(box)
    return labels,boxes ,percents

if __name__ == "__main__": 
    image_path = input('input image path : ')
    img = Image.open(image_path)
    print('please wait minute ... ')
    output =   yolo_runner(image_path)
    labels,boxes ,percents = box_and_label(output)
    for i,(box,label) in enumerate(zip(boxes,labels)): 
        print(i,label,*box)
        # bounding_rect = cv2.boundingRect(tuple(box))
        cropped_image = img.crop(box)
        name = '/home/mahdi/Pictures/preds/pred_box'+str(i)+'.jpg'
        cropped_image.save(name)
