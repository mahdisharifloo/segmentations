# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import argparse
from model import Deeplabv3
from tensorflow import  lite
import tensorflow as tf 

def im_lite_loader(image):
    input_image = np.expand_dims(resized_image, 0)
    return input_image

def lite_feature_ext(input_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    #prediction for input data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="image path")
args = parser.parse_args()

interpreter = tf.lite.Interpreter(model_path='deeplab_lite.tflite')
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

trained_image_width=512 
mean_subtraction_value=127.5

image = np.array(Image.open(args.image_path))
# resize to max dimension of images from training dataset
w, h, _ = image.shape
ratio = float(trained_image_width) / np.max([w, h])
resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
# apply normalization for trained dataset images
resized_image = (resized_image / mean_subtraction_value) - 1.
# pad array to square image to match training images
pad_x = int(trained_image_width - resized_image.shape[0])
pad_y = int(trained_image_width - resized_image.shape[1])
resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
# resized_image = resized_image.astype(np.float32)
resized_image = resized_image.astype(np.float32)
resized_image = np.expand_dims(resized_image, 0)
res = lite_feature_ext(resized_image)

labels = np.argmax(res.squeeze(), -1)



mask = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
result = cv2.bitwise_and(image,image,mask = mask)
# cv2.imwrite('test.jpg',result)

plt.imshow(result)
plt.waitforbuttonpress()
