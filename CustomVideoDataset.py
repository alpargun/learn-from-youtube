
import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu

import torch


logger = getLogger()


class CustomVideoDataset(torch.utils.data.Dataset):
    """ Video dataset. Can be used for image dataset by setting frames as 1. """

    def __init__(
            self,
            data_path,
            frames_per_clip=16,
            frame_step=4,
            num_clips=1,
            filter_long_videos=int(10**9),
            random_clip_sampling=True
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.filter_long_videos = filter_long_videos
        self.random_clip_sampling = random_clip_sampling

        if VideoReader is None:
            raise ImportError('Unable to import "decord", which is required to read videos. For MacOS, check eva-decord')
        
        # Load video paths and labels
        # Load video paths and labels
        data = pd.read_csv(data_path, header=None, delimiter=",")
        self.samples = data.iloc[:,0].to_list()
        self.labels = data.iloc[:,1].to_list() # will be disregarded for SSL training


    def __getitem__(self, idx):

        vid_path = self.data_path

        video = read_video(vid_path)
        label = self.labels[idx]
        
        return video, label

    def __len__(self):
        return len(self.samples)


#%%
data_path = "video_paths.csv"

my_data = CustomVideoDataset(data_path=data_path)
my_data.__len__()
