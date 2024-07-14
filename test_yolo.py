
#%% Imports

import os
import numpy as np
import PIL 
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from glob import glob
import random
import cv2
import warnings
warnings.simplefilter('ignore')

import logging
import warnings

from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from CustomVideoDataset import CustomVideoDataset
from video_transforms import VideoTransform



#%% Load data

logger = logging.getLogger()
#logger.setLevel(logging.INFO)

data_path = "video_paths.csv"

# Load video paths and labels
data = pd.read_csv(data_path, header=None, delimiter=",")
samples = data.iloc[:,0].to_list()
labels = data.iloc[:,1].to_list() # will be disregarded for SSL training
num_samples = len(data)


#%% Define input transform

RESOLUTION = 256

transform = VideoTransform()


#%% Initialize data set and data loader

NUM_FRAMES = 8
SAMPLING_RATE = 8

dataset = CustomVideoDataset(
    data_path=data_path,
    frames_per_clip=NUM_FRAMES, # 8
    frame_step=SAMPLING_RATE, # 8
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    resolution=RESOLUTION,
    transform=transform
)

logger.info('CustomVideoDataset dataset created')

N_EPOCHS = 1
BATCH_SIZE = 1
DEVICE = "cuda"
PIN_MEMORY = True
NUM_WORKERS = 0

data_loader = DataLoader(
    dataset,
    #collate_fn=collator,
    #sampler=dist_sampler,
    batch_size=BATCH_SIZE,
    #drop_last=drop_last,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    #persistent_workers=num_workers > 0,
    shuffle=False
)
logger.info('VideoDataset unsupervised data loader created')


#%% Load and visualize 1 clip from dataset

dir_save = "object-detection/sample-imgs/"
os.makedirs(dir_save, exist_ok=True)

for vid, label, clip_indices, path_video in data_loader:

    video_name = path_video[0].split('/')[-1].split('.')[0]

    for idx, clip in enumerate(vid):
        rows, cols = 2, 4
        fig, axes = plt.subplots(rows,cols)
        for i in range(rows):
            for j in range(cols):
                img = clip[0][i*4+j]  # 0th batch
                img = transform.img_tensor_denormalize(img)

                img = img.numpy().astype(np.uint8)
                axes[i,j].imshow((img))
                plt.axis('off')
                plt.imsave(f'{dir_save}/{video_name}_frame{i*4+j}.jpg', img) # Save the image
        fig.suptitle(f"{path_video}_clip{idx}", fontsize=12)
        fig.show()


#%% Load model

import ultralytics
from ultralytics import YOLO
yolo_model = YOLO('object-detection/yolov8m.pt')


#%% Load image to test YOLO

root_path = 'object-detection/sample-imgs/*'
num_samples = 1
images_data = glob(root_path)

random.seed(1)
random_image = random.sample(images_data, num_samples)

plt.figure(figsize=(10,6))
#for i in range(num_samples):
    #plt.subplot(2,2,i+1)
img = cv2.imread(random_image[i])
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(RGB_img)

# %% Create a list to store the images
yolo_outputs = yolo_model.predict(random_image[i])
output = yolo_outputs[0]
box = output.boxes
names = output.names

for j in range(len(box)):
    labels = names[box.cls[j].item()]
    coordinates = box.xyxy[j].tolist()
    confidence = np.round(box.conf[j].item(), 2)
    #print(f'In this image {len(box)} objects has been detected.')
    print(f'Object {j + 1} is: {labels}')
    print(f'Coordinates are: {coordinates}')
    print(f'Confidence is: {confidence}')
    print('-------')
    
# Store the image in the 'images' list
image = output.plot()[:, :, ::-1]

# %% plotting the images after object detection
print('\n\n-------------------------------------- Images after object detection with YOLOV8 --------------------------------')    

plt.figure(figsize=(10,6))
    #plt.subplot(2, 2, i + 1)
plt.imshow(image)
plt.axis('off')    
plt.tight_layout()
plt.show()


# %% TODO set higher resolution, make text smaller, make boxes moveable to correct the labels.
# plot manually instead of output.plot
