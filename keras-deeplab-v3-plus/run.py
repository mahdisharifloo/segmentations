import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from model import Deeplabv3

# Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# as original image.  Normalization matches MobileNetV2

trained_image_width=512 
mean_subtraction_value=127.5
deeplab_model = Deeplabv3()

image = np.array(Image.open(input('input image file : ')))

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

# make prediction
res = deeplab_model.predict(np.expand_dims(resized_image, 0))
labels = np.argmax(res.squeeze(), -1)


# remove padding and resize back to original image
if pad_x > 0:
    labels = labels[:-pad_x]
if pad_y > 0:
    labels = labels[:, :-pad_y]
labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))

def create_masks(pred_mask):
    copy_mask = np.argmax(pred_mask, axis=-1)
    OUTPUT_CHANNELS = copy_mask.max()
    masks = []
    for chanel in range(OUTPUT_CHANNELS):
        copy_mask[copy_mask==chanel]=255
        masks.append(copy_mask)
        copy_mask = np.argmax(pred_mask, axis=-1)
    return masks

results = create_masks(labels)
for i,result in enumerate(results):
    cv2.imwrite('result{}.jpg'.format(i),result)
    
# plt.imshow(labels)
# plt.waitforbuttonpress()
