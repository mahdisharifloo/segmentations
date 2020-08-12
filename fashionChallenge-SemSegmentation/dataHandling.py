# Create dataset for the fashion problem
# 
# Eduardo Rocha, June 2019
 
import pandas as pd
import numpy as np
import cv2
import json
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from Mask_RCNN.mrcnn import utils, visualize


"""
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import tensorflow as tf
print(tf.__version__)
"""


def resize_image(image_path, image_size):
    """
    Helper function to resize images using opencv
    "image_path"
    "image_size"
    returns resized image
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)  



class CustomDataset(utils.Dataset):
    """
    Based on Matterport base class for dataset
    """


    def __init__(self, df, label_names, data_dir, image_size):
        super().__init__(self)

        self.image_size = image_size
        self.label_names = label_names
        
        # Add classes
        for i, name in enumerate(label_names):
            self.add_class("fashion", i+1, name)
        
        # Add images 
        for i, row in df.iterrows():
            self.add_image("fashion", 
                           image_id=row.name, 
                           path=str(data_dir/'train'/row.name), 
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [self.label_names[int(x)] for x in info['labels']]
    
    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'], self.image_size)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((self.image_size, self.image_size, len(info['annotations'])), dtype=np.uint8)
        labels = []
        
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            
        return mask, np.array(labels)


def MakeDatasets(config, data_visu=False):
    # TODO data visualization
    """
    Loads and pre-process data
    "config": configuration class (config.py)

    returns train and val sets
    """
    
    # labels 
    with open(str(config.DATA_DIR/"label_descriptions.json")) as f:
        label_descriptions = json.load(f)
    label_names = [label['name'] for label in label_descriptions['categories']]

    # segments df
    segment_df = pd.read_csv(config.DATA_DIR/"train.csv")
    segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]

    # Images df
    image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
    size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
    image_df = image_df.join(size_df, on='ImageId')

    # create dataset
    dataset = CustomDataset(image_df, label_names, config.DATA_DIR, config.IMAGE_SIZE)
    dataset.prepare()

    for i in range(6):
        image_id = random.choice(dataset.image_ids)
        print(dataset.image_reference(image_id))
        
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)

    # split dataset
    train_df, valid_df = train_test_split(image_df, test_size=0.20, random_state=42)
    train_dataset = CustomDataset(train_df, label_names, config.DATA_DIR, config.IMAGE_SIZE)
    train_dataset.prepare()
    valid_dataset = CustomDataset(valid_df, label_names, config.DATA_DIR, config.IMAGE_SIZE)
    valid_dataset.prepare()

    return train_dataset, valid_dataset