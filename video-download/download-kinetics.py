
#%%
import datetime
import json
import random

import pandas as pd

from video_downloader import download_video


#%% Read kinetics partition csv file

path_test = "../k400/test.csv"

df_test = pd.read_csv(path_test)
df_test

#%% Randomly choose number of videos

NUM_VIDEOS = 5

vid_indices = random.choices(df_test.index, k=NUM_VIDEOS)

df_selected = df_test.iloc[vid_indices]
df_selected


#%% Get kinetics ids

json_filename = "../k400/kinetics_classnames.json"

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")


# %% Download videos

BASE_DIR = "../k400/videos/test"

for idx, row in df_selected.iterrows():
    video_id = row["youtube_id"]
    label = row["label"]
    time_start = row["time_start"] # in seconds
    time_end = row["time_end"] # in seconds

    #label = label.replace(" ", "-")

    id = label

    for k,v in kinetics_id_to_classname.items():
        if v == label:
            print(k,v)
            id = k

    download_dir = f"{BASE_DIR}/{id}/"

    # convert seconds "to HH:MM:SS" format for FFMPEG
    time_start = str(datetime.timedelta(seconds=time_start))
    time_end = str(datetime.timedelta(seconds=time_end))
    download_video(video_id=video_id, download_dir=download_dir, time_start=time_start, time_end=time_end)

# %%
