
import numbers
import os
import pathlib
import warnings

from logging import getLogger

import cv2
import numpy as np
import pandas as pd
import torch

from decord import VideoReader, cpu


logger = getLogger()
rng = np.random.default_rng(seed=5)


class CustomVideoDataset(torch.utils.data.Dataset):
    """ Video dataset. Can be used for image dataset by setting frames as 1. """

    def __init__(
            self,
            data_path,
            frames_per_clip=16,
            frame_step=4,
            num_clips=1,
            filter_long_videos=int(10**9),
            random_clip_sampling=True,
            allow_clip_overlap=False,
            resolution=224,
            transform=None
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.filter_long_videos = filter_long_videos
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.resolution = resolution
        self.transform = transform

        if VideoReader is None:
            raise ImportError('Unable to import "decord", which is required to read videos. For MacOS, check eva-decord')
        
        # Load video paths and labels
        # Load video paths and labels
        data = pd.read_csv(data_path, header=None, delimiter=",")
        self.samples = data.iloc[:,0].to_list()
        self.labels = data.iloc[:,1].to_list() # will be disregarded for SSL training


    def __getitem__(self, index):

        sample = self.samples[index]

        # Keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            buffer, clip_indices = self.load_video_decord(sample)  # [T H W 3]
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = rng.integers(self.__len__())
                sample = self.samples[index]

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]
        
        buffer = split_into_clips(buffer)
        
        #buffer = [self.resize_clip(clip, (self.resolution, self.resolution)) for clip in buffer]
        
        buffer = [np.stack(clip) for clip in buffer] # T C H W for each clip

        buffer = torch.tensor(buffer, dtype=torch.float32) #torch.from_numpy(buffer).float()
        
        if self.transform:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices, sample


    def load_video_decord(self, sample):

        """ Load video content using Decord """

        fname = sample
        print("video path: ", sample)

        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')

        #if _fsize > filter_long_videos:
            #warnings.warn(f'skipping long video of size {_fsize=} (bytes)')

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            print("failed to read video")

        print("avg fps of video: ", vr.get_avg_fps())

        total_frames = len(vr)
        duration = None # 10 # 10 sec clips
        fpc = self.frames_per_clip
        fstp = self.frame_step
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

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = rng.integers(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
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
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()

        return buffer, clip_indices

    def resize_clip(self, clip, size, interpolation='bilinear'):

        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                and im_h == size):
                return clip
            new_h, new_w = self.get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
            
        else:
            size = size[0], size[1]

        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
                
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
        return scaled

        
    def get_resize_sizes(self, im_h, im_w, size):

        if im_w < im_h:
            ow = size
            oh = int(size * im_h / im_w)
        else:
            oh = size
            ow = int(size * im_w / im_h)
        return oh, ow


    def __len__(self):
        return len(self.samples)
    

#%%
data_path = "video_paths.csv"

my_data = CustomVideoDataset(data_path=data_path)
my_data.__len__()
