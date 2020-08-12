# -*- coding: utf-8 -*-

from tensorflow.keras.applications import mobilenet_v2
import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 
import pix2pix
from tensorflow.keras.utils import plot_model
import argparse
import cv2

def image_preprocess(im_path):
    image = cv2.imread(im_path )
    input_image = cv2.resize(image,(128,128))
    input_image = np.expand_dims(input_image,axis=0)
    return input_image

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128,128, 3])
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
def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = np.multiply(pred_mask,100)
    # pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def show_predictions(sample_image, num=1):
    pred = model.predict(sample_image)[0]
    result = create_mask(pred)
    cv2.imwrite('result.jpg',result)
    # cv2.imshow('input image',sample_image[0])
    # cv2.imshow('input image',sample_image)
    # cv2.imshow('result',result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
     
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

OUTPUT_CHANNELS = 3 

base_model = mobilenet_v2.MobileNetV2(input_shape=(128,128,3),include_top=False) 

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="image path")
args = parser.parse_args()


# im_path = '/home/mahdi/projects/dgkala/data/watch/105445894.jpg'
sample_image = image_preprocess(args.image_path)
show_predictions(sample_image)

# class DisplayCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     # clear_output(wait=True)
#     show_predictions()
#     print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
    

# EPOCHS = 20
# VAL_SUBSPLITS = 5
# BATCH_SIZE = 64
# BUFFER_SIZE = 1000

# TRAIN_LENGTH = info.splits['train'].num_examples
# STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE



# VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

# model_history = model.fit(train_dataset, epochs=EPOCHS,
#                           steps_per_epoch=STEPS_PER_EPOCH,
#                           validation_steps=VALIDATION_STEPS,
#                           validation_data=test_dataset,
#                           callbacks=[DisplayCallback()])







