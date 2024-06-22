# -*- coding: utf-8 -*-
"""
Created on Mon May  6 23:36:42 2024

@author: Med Aziz Bouzid
"""

#%%
import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model
from torchvision import transforms
import imgaug.augmenters as iaa
from tensorflow.keras.optimizers import Adam
import random


#%%
annotations_dir="C:/Users/azizb/Desktop/Mask-RCNN-TF2-master/NWPU/VHR10/ground truth"
class NwpuDataset(mrcnn.utils.Dataset):
    def __init__(self, mode='training'):
        super().__init__()
        self.mode = mode
        self.annotations_dir = annotations_dir


    # def load_and_split_dataset(dataset_dir, split_ratio=0.7):
    #     # Construction de la liste des objets avec leurs annotations correspondantes
    #     object_list = []
    #     annotations_dir = os.path.join(dataset_dir, 'ground truth')
    #     for filename in os.listdir(annotations_dir):
    #         if filename.endswith('.txt'):
    #             image_id = filename.split('.')[0]
    #             img_filename = f"{image_id}.jpg"
    #             img_path = os.path.join(dataset_dir, 'positive image set', img_filename)
    #             ann_path = os.path.join(annotations_dir, filename)
    #             with open(ann_path, 'r') as file:
    #                 content = file.readlines()
    #                 if not content:
    #                     continue
    #                 for line in content:
    #                     matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*(\d+)', line)
    #                     for match in matches:
    #                         x1, y1, x2, y2, class_id = map(int, match)
    #                         object_list.append({'image_id': image_id, 'img_path': img_path, 'ann_path': ann_path,
    #                                             'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class_id': class_id})
    
    #     # Division de la liste des objets en ensembles d'entraînement et de test
    #     random.shuffle(object_list)
    #     split_index = int(len(object_list) * split_ratio)
    #     train_objects = object_list[:split_index]
    #     test_objects = object_list[split_index:]
    
    #     return train_objects, test_objects
    
    def load_dataset(self, dataset_dir, is_train=True):
      self.add_class("dataset", 1, "airplane")
      self.add_class("dataset", 2, "ship")
      self.add_class("dataset", 3, "storage tank")
      self.add_class("dataset", 4, "baseball diamond")
      self.add_class("dataset", 5, "tennis court")
      self.add_class("dataset", 6, "basketball court")
      self.add_class("dataset", 7, "ground track field")
      self.add_class("dataset", 8, "harbor")
      self.add_class("dataset", 9, "bridge")
      self.add_class("dataset", 10, "vehicle")
      images_dir = os.path.join(dataset_dir, 'positive image set')
      annotations_dir = os.path.join(dataset_dir, 'ground truth')
    
      # List all annotation files
      annotation_files = os.listdir(annotations_dir)
    
      # Split into training and testing
      split = int(0.7 * len(annotation_files))  # 70% for training
      train_files = annotation_files[:split]
      test_files = annotation_files[split:]
    
      for ann_filename in annotation_files:
          img_id = int(ann_filename[:-4])
          img_filename = f"{img_id:03d}.jpg"
    
          img_path = os.path.join(images_dir, img_filename)
          ann_path = os.path.join(annotations_dir, ann_filename)
    
          # Check if annotation file exists
          if not os.path.exists(ann_path):
              print(f"Annotation file {ann_filename} does not exist.")
              continue
    
          # Read and check annotations
          with open(ann_path, 'r') as file:
              content = file.readlines()
              if not content:
                  print(f"Annotation file {ann_filename} is empty.")
                  continue
    
          # Add objects to dataset
          for line in content:
              matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*(\d+)', line)
              for match in matches:
                  x1, y1, x2, y2, class_id = map(int, match)
                  object_id = f"{img_id}_{match}"  # Unique object ID
    
                  if is_train and ann_filename in train_files:
                      self.add_image('dataset', image_id=object_id, path=img_path, annotation=[x1, y1, x2, y2, class_id])
                      print(f'Added to Train: {img_filename} - Object {object_id}')
                  elif not is_train and ann_filename in test_files:
                      self.add_image('dataset', image_id=object_id, path=img_path, annotation=[x1, y1, x2, y2, class_id])
                      print(f'Added to Test: {img_filename} - Object {object_id}')
    
    def load_mask(self, image_id):
    
        # Access the image information using the image ID
        info = self.image_info[image_id]
    
        # Extract the annotation file path from the image information
        filename = info['annotation']
    
        # Construct the annotation file path
        path = os.path.join(self.annotations_dir, filename)
    
        # Print the path for verification
        print(path)
        boxes = self.extract_boxes(path)
    
        # Open the image to get its size
        image = Image.open(info['path'])
        width, height = image.size
    
        # Update dimensions after resizing
        masks = np.zeros([height, width, len(boxes)], dtype=np.uint8)
        class_ids = []
    
        for i, (x1, y1, x2, y2, class_id) in enumerate(boxes):
            masks[y1:y2, x1:x2, i] = 1
            class_ids.append(class_id)
    
        return masks, np.array(class_ids)
    
    def extract_boxes(self, filename):
        boxes = []
        if not isinstance(filename, str):
            filename = str(filename)
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*(\d+)', line)
                for match in matches:
                    x1, y1, x2, y2, class_id = map(int, match)
                    boxes.append((x1, y1, x2, y2, class_id))
        return boxes


     


#%%
import matplotlib.pyplot as plt
import cv2

# Load the image using OpenCV
image = cv2.imread("210.jpg")

# Create an instance of the dataset
dataset_train = NwpuDataset()

# Apply augmentation
augmented_image = dataset_train.augment_data(image)

# Plot original and augmented images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Augmented Image')
axes[1].axis('off')
plt.show()

#%%

#     # Chargement des données et division en ensembles d'entraînement et de test
#     train_objects, test_objects = NwpuDataset.load_and_split_dataset('VHR10')

#     # Construction des ensembles d'entraînement et de test
#     dataset_train = NwpuDataset(mode='training')
#     dataset_train.add_objects(train_objects)
    
#     dataset_test = NwpuDataset(mode='testing')
#     dataset_test.add_objects(test_objects)
# print("Nombre d'images dans l'ensemble d'entraînement :", len(dataset_train.image_info))

dataset_train = NwpuDataset(mode='training')
dataset_train.load_dataset(dataset_dir='VHR10', is_train=True)
dataset_train.prepare()

dataset_test = NwpuDataset(mode='testing')
dataset_test.load_dataset(dataset_dir='VHR10', is_train=False)
dataset_test.prepare()

#%%

class NwpuConfig(mrcnn.config.Config):
    NAME = "kangaroo_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 11

    STEPS_PER_EPOCH = 32

    # Learning rate
    LEARNING_RATE = 0.001

# Model Configuration
nwpu_config = NwpuConfig()

#%%
# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=nwpu_config)

#%%
# Compile the model without specifying the optimizer
#optimizer = Adam(lr=nwpu_config.LEARNING_RATE)  # Corrected import
#model.compile(optimizer=optimizer, loss="mean_squared_error")

#%%
# Load pre-trained weights (COCO)
model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

#%%

# Train the model
model.train(train_dataset=dataset_train, 
            val_dataset=dataset_test,
            learning_rate=nwpu_config.LEARNING_RATE, 
            epochs=15,
            layers='heads')

#%%
# Save trained model weights
model_path = 'nwpu_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)


#%%
import time
from sklearn.metrics import precision_recall_curve, average_precision_score
def compute_ap(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores, iou_threshold=0.5):
    """Compute Average Precision for a single class"""
    # Initialize true and false positive counts
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    # Initialize IoU list
    ious = np.zeros((len(pred_boxes), len(gt_boxes)))
    # Loop over predictions and match with ground truth
    for i, pred_box in enumerate(pred_boxes):
        # Compute IoU with ground truth boxes
        for j, gt_box in enumerate(gt_boxes):
            ious[i, j] = compute_iou(pred_box, gt_box)
        # Find the best match
        best_match_idx = np.argmax(ious[i])
        best_iou = ious[i, best_match_idx]
        # Assign detection as true positive/don't care/false positive
        if best_iou >= iou_threshold and gt_class_ids[best_match_idx] == pred_class_ids[i]:
            if tp[best_match_idx] == 0:
                tp[i] = 1  # True positive
                tp[best_match_idx] = 1
            else:
                fp[i] = 1  # False positive (multiple detections)
        else:
            fp[i] = 1  # False positive
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    # Compute average precision
    ap = average_precision_score(tp, precisions)
    return ap

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) for two bounding boxes"""
    # Extract coordinates of intersection area
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])
    # Compute intersection area
    intersection_area = max(0, x2_int - x1_int + 1) * max(0, y2_int - y1_int + 1)
    # Compute areas of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # Compute union area
    union_area = box1_area + box2_area - intersection_area
    # Compute IoU
    iou = intersection_area / union_area
    return iou


#%%
def evaluate_model(model, dataset):
    start_time = time.time()
    APs = []
    IoUs = []
    for image_id in dataset.image_ids:
        image, _, gt_class_id, gt_bbox, _ = mrcnn.utils.load_image_gt(dataset, nwpu_config, image_id, use_mini_mask=False)
        molded_image = np.expand_dims(mrcnn.utils.mold_image(image, nwpu_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP = compute_ap(gt_bbox, gt_class_id, r['rois'], r['class_ids'], r['scores'])
        APs.append(AP)
        # Compute IoU
        for i, gt_box in enumerate(gt_bbox):
            iou = compute_iou(gt_box, r['rois'][i])
            IoUs.append(iou)
    mean_ap = np.mean(APs)
    mean_iou = np.mean(IoUs)
    end_time = time.time()
    computation_time = end_time - start_time
    return mean_ap, mean_iou, computation_time
#%%
# Evaluate the model
mean_ap, mean_iou, computation_time = evaluate_model(model, dataset_test)
print("Mean Average Precision (mAP):", mean_ap)
print("Mean Intersection over Union (mIoU):", mean_iou)
print("Computation Time:", computation_time)


#%%
# Precision-Recall Curve
def plot_pr_curve(precision, recall):
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    
    
#%%    
# Compute precision and recall
precision, recall, _ = precision_recall_curve(gt_class_ids, gt_scores)

#%%   
# Plot Precision-Recall Curve
plot_pr_curve(precision, recall)

#%%
# Define learning rates and epochs for each fine-tuning strategy
#learning_rates = [0.1, 0.01, 0.001]
#epochs = [30, 30, 400]

# Fine-tuning strategy 1: Train the head layer for 30 epochs
#model.train(train_dataset=dataset_train, 
#            val_dataset=dataset_test,
#            learning_rate=learning_rates[0], 
#            epochs=epochs[0], 
#            layers='heads')

# Fine-tuning strategy 2: Train certain convolution layers for 30 epochs each
#model.train(train_dataset=dataset_train, 
#            val_dataset=dataset_test,
#            learning_rate=learning_rates[1], 
#            epochs=epochs[1], 
#            layers='3+')  # Train from layer 3 onwards

#model.train(train_dataset=dataset_train, 
#            val_dataset=dataset_test,
#            learning_rate=learning_rates[2], 
#            epochs=epochs[2], 
#            layers='4+')  # Train from layer 4 onwards

# Fine-tuning strategy 3: Train deeper convolution layers for 400 epochs
#model.train(train_dataset=dataset_train, 
#            val_dataset=dataset_test,
#            learning_rate=learning_rates[2], 
#            epochs=epochs[2], 
#            layers='5+')  # Train from layer 5 onwards