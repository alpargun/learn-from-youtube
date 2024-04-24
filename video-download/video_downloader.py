
#%% Imports

import datetime
import os
import subprocess
from pytube import YouTube
from pytube.exceptions import VideoUnavailable, VideoPrivate


def download_video(video_id, download_dir, time_start=None, time_end=None):

    BASE_URL = "http://youtube.com/watch?v="
    video_url = BASE_URL + video_id

    try:
        yt = YouTube(video_url)

        # Show video details
        print(yt.title)
        print(yt.description)
        print(str(datetime.timedelta(seconds=yt.length)))

        # Download the video
        os.makedirs(download_dir, exist_ok=True)

        temp_video_name = video_id + "_temp.mp4"

        yt.streams.filter(progressive=True, file_extension='mp4', resolution="360p").first().download(output_path=download_dir, filename=temp_video_name) # only_video=True results in double the duration of video

        # Remove audio

        input_video = download_dir + temp_video_name
        output_video = download_dir + video_id + ".mp4"

        if time_start is not None and time_end is not None:
            subprocess.call(
                f"ffmpeg -i {input_video} -ss {time_start} -to {time_end} -an -c:v copy {output_video} -y", shell=True
            )
        else:
            subprocess.call(
                f"ffmpeg -i {input_video} -an -c:v copy {output_video} -y", shell=True
            )

        os.remove(input_video)
        
    except VideoPrivate:
        print(f'Video {video_url} is private, skipping.')
    except VideoUnavailable:
        print(f'Video {video_url} is unavaialable, skipping.')
    
   
