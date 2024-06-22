# -*- coding: utf-8 -*-
"""
Created on Mon May  6 23:19:35 2024

@author: Med Aziz Bouzid
"""

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

# Update the class names according to your dataset
CLASS_NAMES = ['BG', 'airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 
               'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']

class CustomConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "custom_inference"
    
    # Set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes = number of classes + 1 (for background)
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and load the weights
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=CustomConfig(),
                             model_dir=os.getcwd())

# Load the trained weights
model.load_weights(filepath="nwpu_mask_rcnn_trained.h5", 
                   by_name=True)

# Load the input image and convert it from BGR to RGB channel
image = cv2.imread("210.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image
r = r[0]

# Visualize the detected objects
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
