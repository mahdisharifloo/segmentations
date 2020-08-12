# Training
# 
# Eduardo Rocha, June 2019

# TODOs:
#   Test Set

import Mask_RCNN.mrcnn.model as modellib
from imgaug import augmenters as iaa
import numpy as np
import os
import time

import config
import dataHandling

import tensorflow as tf
print(tf.__version__) 
# docker: nvcr.io/nvidia/tensorflow:19.05-py3 -->> 1.13.1
# Nvidia DGX workstation

# time tracking [start, head training, fine-tune]
time_tracker = [time.time()]

# Load configuration
config = config.CustomConfigCOCO()

# datasets
# TODO: define split ratio in config
train_dataset, valid_dataset = dataHandling.MakeDatasets(config)

model = modellib.MaskRCNN(mode='training', config=config, model_dir=str(config.ROOT_DIR))

model.load_weights(config.WEIGHTS_PATH, by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

# Step 1 train heads
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5), # only horizontal flip here
    iaa.Crop(percent=(0, 0.1)),
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.25))
    ),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)
model.train(train_dataset, valid_dataset,
            learning_rate=config.LEARNING_RATE,
            epochs=config.EPOCHS_1,
            layers='heads', # train only heads, freeze rest
            augmentation=augmentation)
history = model.keras_model.history.history

# time tracking
time_tracker.append(time.time())
print("\nHead training time: %d min\n"%\
    (int((time_tracker[1]-time_tracker[0])/60)))

# Step 2 fine tune all network
model.train(train_dataset, valid_dataset,
            learning_rate=config.LEARNING_RATE/10,
            epochs=config.EPOCHS_2,
            layers='all',
            augmentation=augmentation)
new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

# time tracking
time_tracker.append(time.time())
print("\nHead training time: %d min"%\
    (int((time_tracker[1]-time_tracker[0])/60)))
print("Fine-tuning time: %d min"%\
    (int((time_tracker[2]-time_tracker[1])/60)))
print("Total training time: %d min\n"%\
    (int((time_tracker[2]-time_tracker[0])/60)))

# loss
best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])
print("Losses:\n")
print(history['loss'])
print("")
print(history['val_loss'])
print("")
print(history['mrcnn_class_loss'])
print("")
print(history['val_mrcnn_class_loss'])
print("")
print(history['mrcnn_mask_loss'])
print("")
print(history['val_mrcnn_mask_loss'])
print("")