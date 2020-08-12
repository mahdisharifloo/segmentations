# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 
import pix2pix
from tensorflow.keras.utils import plot_model
import argparse
import cv2
from tensorflow.keras.preprocessing import image

def image_preprocess(im_path):
    image = cv2.imread(im_path )
    input_image = cv2.resize(image,(224,224))
    input_image = np.expand_dims(input_image,axis=0)
    return input_image

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[224,224, 3])
  x = inputs
  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])
  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])
  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

# plot_model(model, show_shapes=True)
def create_masks(pred_mask):
    copy_mask = np.argmax(pred_mask, axis=-1)
    masks = []
    for chanel in range(OUTPUT_CHANNELS):
        copy_mask[copy_mask==chanel]=255
        masks.append(copy_mask)
        copy_mask = np.argmax(pred_mask, axis=-1)
    # pred_mask = np.multiply(pred_mask,255)
    # pred_mask = pred_mask[..., tf.newaxis]
    return masks

def show_predictions(sample_image, num=1):
    pred = model.predict(sample_image)[0]
    results = create_masks(pred)
    for i,result in enumerate(results):
        cv2.imwrite('result{}.jpg'.format(i),result)
    # cv2.imshow('input image',sample_image[0])
    # cv2.imshow('input image',sample_image)
    # cv2.imshow('result',result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
     
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

OUTPUT_CHANNELS = 7

# base_model = mobilenet_v2.MobileNetV2(input_shape=(128,128,3),include_top=False) 
base_model = load_model('/home/mahdi/projects/dgkala/models/fashoin_model.h5')

# fashion_list = ['pants', 'school_bag', 'sneaker', 'socks', 't-shirt', 'watch', 'woman bag']
# layer_name_list = []
# for layer in  base_model.layers:
#     layer_name_list.append(layer.name)
    
layer_names = [
    'activation_50',   # 112*112
    'activation_59',   # 56*56
    'activation_71',   # 28*28
    'activation_89',   # 14*14
    'activation_98',   # 7*7
]

layers = [base_model.get_layer(name).output for name in layer_names]
# you have to see layers and select activations layers that you need them \
#   for feature extraction.
# Use the activations of these layers

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(2048, 3),  # 7*7 -> 14*14
    pix2pix.upsample(1024, 3),  # 14*14 -> 28*28
    pix2pix.upsample(512, 3),  # 28*28-> 56*56
    pix2pix.upsample(256, 3),   # 56*56 -> 112*112
]


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="image path")
args = parser.parse_args()


im_path = '/home/mahdi/projects/dgkala/data/watch/105445894.jpg'
sample_image = image_preprocess(args.image_path)

# encoder_pred = down_stack.predict(sample_image)
show_predictions(sample_image)




