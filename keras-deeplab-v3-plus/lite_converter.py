# -*- coding: utf-8 -*-

from tensorflow import  lite
from model import Deeplabv3


deeplab_model = Deeplabv3()


converter = lite.TFLiteConverter.from_keras_model(deeplab_model)
tf_lite_model =  converter.convert()

open('deeplab_lite.tflite','wb').write(tf_lite_model)
