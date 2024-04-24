
#%%
import json
import logging

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from CustomVideoDataset import CustomVideoDataset
from video_transforms import VideoTransform

logging.basicConfig()

from torch.utils.tensorboard import SummaryWriter


#%% Load the model:

# Choose the `slow_r50` model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)


#%% Get Kinetics class names from json file

json_filename = "k400/kinetics_classnames.json"

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")


#%% Load data

logger = logging.getLogger()
#logger.setLevel(logging.INFO)

data_path = "k400/k400_paths.csv"

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

#vid, label, clip_indices, path_video = next(iter(data_loader))

for vid, label, clip_indices, path_video in data_loader:

    for idx, clip in enumerate(vid):
        rows, cols = 2, 4
        fig, axes = plt.subplots(rows,cols)
        fig.suptitle(f"{path_video}_clip{idx}", fontsize=16)
        for i in range(rows):
            for j in range(cols):
                img = clip[0][i*4+j].numpy().astype(np.uint8)
                axes[i,j].imshow((img)) # 0th batch
        fig.show()

        # Pass the input clip through the model
        inputs = clip.squeeze()
        print("Shape inputs: ", inputs.shape)
        #inputs = inputs[0:8,...]
        inputs = inputs.permute(3, 0, 1, 2) # C T H W

        #inputs = inputs / 255.0  # .float() # / 255.0

        preds = model(inputs[None ,...])

        # Get the predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=5).indices[0]
        print(preds.topk(k=5))

        print(f"\nLabel: {kinetics_id_to_classname[label.item()]} \t| Prediction: {kinetics_id_to_classname[pred_classes[0].item()]}")

        # Map the predicted classes to the label names
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
        print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))


#%%