# Challenge

[iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6):
Fine-grained segmentation task for fashion and apparel

### Task

"We present a new clothing dataset with the goal of introducing a novel fine-grained segmentation task by joining forces between the fashion and computer vision communities. The proposed task unifies both categorization and segmentation of rich and complete apparel attributes, an important step toward real-world applications."

# Approach

[Mask R-CNN (paper)](https://arxiv.org/abs/1703.06870) was chosen because it is currently among the best methods for this task ([top 10 in Instance Segmentation on COCO](https://paperswithcode.com/sota/instance-segmentation-on-coco)), and it is fairly simple to be implemented.

Since the dataset provided is not huge (45625 images), and to save training time, it was decided to use transfer learning. The model was based on the [implementation by matterport](https://github.com/matterport/Mask_RCNN) because they already provide a pre-trained model based on Feature Pyramid Network (FPN) and a ResNet101 backbone. The implementation is made in TensorFlow/Keras.

Since less than 4% of the segments have attributes, these attributes were ignored as an initial approach for simplification.

# Nvidia DGX Station

A Nvidia DGX Station was used for training. It consists of a system with 4 32gb Tesla V100 GPUs.

The communication with the station was made by SSH, so the script [DGX_ssh_comm.py](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/DGX_ssh_comm.py) was made in order to facilitate the file transfer and automatic execution of the training during development and debugging. 

A docker container was built by adding the requirements for the Matterport implementation to the Nvidia container  nvcr.io/nvidia/tensorflow:19.05-py3.

# Training 

### 1) First look at the data

* 45625 images, 330000 sements, less than 4% of segments have attributes. Ignore attributes initially.
* Objects have consistent vertical orientation (e.g. shoes usually have the laces on top) so it was decided to use only horizontal flip for aumentation.
* Also shifting and scaling for aumentation.

### 2) Load and split dataset

* Training Set: 80% for training, 20% for validation.
* Testing Set: as it is provided by Kaggle.

* [dataHandling.py](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/dataHandling.py)

### 3) Training parameters

The training parameters are defined in [config.py](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/config.py)

Some of the used parameters:
```python
class CustomConfigCOCO(Config):
    """
    My custom class using COCO pre-trained weights
    """
    # training
    LEARNING_RATE = 2e-4
    # train in three stages - 3 6 8
    EPOCHS_1 = 10 # head only
    EPOCHS_2 = 15 + EPOCHS_1 # Finetune all
    # instances / (GPU_COUNT * IMAGES_PER_GPU) train = 4*9125, val = 9125
    STEPS_PER_EPOCH = 1520
    VALIDATION_STEPS = 380
    # DGX workstation
    GPU_COUNT = 4
    IMAGES_PER_GPU = 6 #(32Gb/GPU)
    # ...
```

**NOTE: the model was trained only once. These parameters haven't been tuned, they are probably not the ideal set.**

### 4) Training

The training was divided in two parts. First, only the head of the network was trained. Then, the entire network was fine-tuned. Training took 7h14min.

```python
# Step 1 train heads
model.train(train_dataset, valid_dataset,
            learning_rate=config.LEARNING_RATE,
            epochs=config.EPOCHS_1,
            layers='heads', # train only heads, freeze rest
            augmentation=augmentation)
history = model.keras_model.history.history

# Step 2 fine tune all network
model.train(train_dataset, valid_dataset,
            learning_rate=config.LEARNING_RATE/10,
            epochs=config.EPOCHS_2,
            layers='all',
            augmentation=augmentation)
```

Loss:
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/loss.jpg)

Apparently, the network overfits the training set a little bit after 17~18 epochs. Therefore, the model after 17 epochs was chosen as the final model.

# Results

Some of the results from the test set:
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result1.jpg)
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result2.jpg)
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result3.jpg)
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result4.jpg)
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result5.jpg)
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result6.jpg)
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result7.jpg)
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result8.jpg)
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result9.jpg)
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/result10.jpg)




# Next steps

* Tune training parameters in order to get a better model. 
* Take advantage of the fact that the position of clothing items is related to a specific body part. There is a lot of models that can detect humans and segment its body parts. A possible approach would be to use a pretrained model based on Feature Pyramid Network (FPN) and a ResNet101 backbone (such as the one used here) for feature extraction, add the detected body parts positions (detected by a different model) to the feature vector, and, finally, add the head of the model. That is:
![](https://github.com/mcreduardo/fashionChallenge-SemSegmentation/blob/master/images/schem.jpg)
As seen in the results above, many of the wrong detections would be solved. The results are expected to get better. Note that this model would increase the training and prediction time significantly, but, given the fact that the provided dataset is fairly small, this might be a good solution. 
* Test other models and architectures.
* Ensemble various different models. Since these type of competitions don't require real-time prediction, that is, the computational cost for predictions is not an issue, the use of ensembles are very popular.