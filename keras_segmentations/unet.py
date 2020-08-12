# -*- coding: utf-8 -*-

from segmentation_models import Unet,FPN,Linknet,PSPNet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from tensorflow.keras.datasets import mnist


BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)




# load your data
(x_train, y_train),( x_val, y_val) = mnist.load_data()

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)
# define model
model = Unet(BACKBONE,encoder_weights=None,input_shape=(28,28,3))
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

# fit model
model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=100,
    validation_data=(x_val, y_val),
)