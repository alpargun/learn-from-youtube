# Script to test video downloader

#%% Imports

import datetime
import os
import subprocess
from pytube import YouTube


#%%

BASE_URL = "http://youtube.com/watch?v="
video_id =  "u3_2gAuPjxM" # "_VwmNdw5BYc" # "eR5vsN1Lq4E"
video_url = BASE_URL + video_id

yt = YouTube(video_url)

#%% Create stream object

stream = yt.streams.filter(progressive=True, file_extension='mp4', resolution="360p").first() # only_video=True results in double the duration of video

#%% Show video details

print(yt.title)
print(yt.description) # if called before stream, returns null
print(str(datetime.timedelta(seconds=yt.length)))

#%% Download the video

download_dir = "videos/driving/"
os.makedirs(download_dir, exist_ok=True)

temp_video_name = video_id + "_temp.mp4"

stream.download(output_path=download_dir, filename=temp_video_name)


#%% Remove audio

input_video = download_dir + temp_video_name
output_video = download_dir + video_id + ".mp4"

command = f"ffmpeg -i {input_video} -an -c:v copy output.mp4"
subprocess.call(
    f"ffmpeg -i {input_video} -an -c:v copy {output_video} -y", shell=True
)

os.remove(input_video)

