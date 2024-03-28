
#%% Import statements

import os
import pathlib
import warnings

from logging import getLogger

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from decord import VideoReader, cpu

import torch


logger = getLogger()

#%%
rng = np.random.default_rng(seed=5)

data_path = "video_paths.csv"

# Load video paths and labels
data = pd.read_csv(data_path, header=None, delimiter=",")
samples = data.iloc[:,0].to_list()
labels = data.iloc[:,1].to_list() # will be disregarded for SSL training
num_samples = len(data)

# %%

""" Load video content using Decord """

# dummy assign variables
sample = samples[0]
frames_per_clip = 16
frame_step=4
num_clips=1
random_clip_sampling=True
allow_clip_overlap=False
#filter_long_videos=int(10**9)
# ---------------------------------------

fname = sample
if not os.path.exists(fname):
    warnings.warn(f'video path not found {fname=}')

_fsize = os.path.getsize(fname)
if _fsize < 1 * 1024:  # avoid hanging issue
    warnings.warn(f'video too short {fname=}')

#if _fsize > filter_long_videos:
    #warnings.warn(f'skipping long video of size {_fsize=} (bytes)')

#%%
try:
    vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
except Exception:
    print("failed to read video")

print("avg fps of video: ", vr.get_avg_fps())

#%%
total_frames = len(vr)
duration = None # 10 # 10 sec clips
fpc = frames_per_clip
fstp = frame_step
if duration is not None:
    try:
        fps = vr.get_avg_fps()
        fstp = int(duration * fps / fpc)
    except Exception as e:
        warnings.warn(e)
clip_len = int(fpc * fstp)

if len(vr) < clip_len:
    warnings.warn(f'skipping video of length {len(vr)}')

vr.seek(0)  # Go to start of video before sampling frames

#%%
# Partition video into equal sized segments and sample each clip
# from a different segment
partition_len = len(vr) // num_clips

all_indices, clip_indices = [], []
for i in range(num_clips):

    if partition_len > clip_len:
        # If partition_len > clip len, then sample a random window of
        # clip_len frames within the segment
        end_indx = clip_len
        if random_clip_sampling:
            end_indx = rng.integers(clip_len, partition_len) #np.random.randint(clip_len, partition_len)
        start_indx = end_indx - clip_len
        indices = np.linspace(start_indx, end_indx, num=fpc)
        indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
        # --
        indices = indices + i * partition_len
    else:
        # If partition overlap not allowed and partition_len < clip_len
        # then repeatedly append the last frame in the segment until
        # we reach the desired clip length
        if not allow_clip_overlap:
            indices = np.linspace(0, partition_len, num=partition_len // fstp)
            indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
            indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
            # --
            indices = indices + i * partition_len

        # If partition overlap is allowed and partition_len < clip_len
        # then start_indx of segment i+1 will lie within segment i
        else:
            sample_len = min(clip_len, len(vr)) - 1
            indices = np.linspace(0, sample_len, num=sample_len // fstp)
            indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
            indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
            # --
            clip_step = 0
            if len(vr) > clip_len:
                clip_step = (len(vr) - clip_len) // (num_clips - 1)
            indices = indices + i * clip_step

    clip_indices.append(indices)
    all_indices.extend(list(indices))

buffer = vr.get_batch(all_indices).asnumpy()
buffer, clip_indices
print("clip indices: ", clip_indices)


# %% Display all frames in the clip

for idx, val in enumerate(buffer):
    plt.figure(1); plt.clf()
    plt.imshow(val)
    plt.title('Frame ' + str(idx))
    plt.pause(0.2)

# %% Show all frames in one figure

fig, axes = plt.subplots(4,4)
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(buffer[i*4+j])
fig.show()

# %%
