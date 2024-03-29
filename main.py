
#%%
import logging
import os
import pathlib
import warnings

from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from CustomVideoDataset import CustomVideoDataset


logging.basicConfig()

from torch.utils.tensorboard import SummaryWriter


#%% Load data

logger = logging.getLogger()
#logger.setLevel(logging.INFO)

data_path = "video_paths.csv"

# Load video paths and labels
data = pd.read_csv(data_path, header=None, delimiter=",")
samples = data.iloc[:,0].to_list()
labels = data.iloc[:,1].to_list() # will be disregarded for SSL training
num_samples = len(data)

dataset = CustomVideoDataset(
    data_path=data_path,
    frames_per_clip=16,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False
)

logger.info('CustomVideoDataset dataset created')

#dist_sampler = torch.utils.data.distributed.DistributedSampler(
    #dataset,
    #num_replicas=world_size,
    #rank=rank,
    #shuffle=True)

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


#%% Training and Validation

for epoch in range(N_EPOCHS):

    print("training...")

    with tqdm(data_loader, unit="batch") as tepoch: # show progress bar for batches
        for idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")

            vid, label, clip_indices, path_video = batch
            print("clip indices: ", clip_indices)
            print(f"frame shape: ", vid[0].shape)
            fig, axes = plt.subplots(4,4)
            fig.suptitle(path_video, fontsize=16)
            for i in range(4):
                for j in range(4):
                    axes[i,j].imshow(vid[i*4+j].squeeze())
            fig.show()


# %%
