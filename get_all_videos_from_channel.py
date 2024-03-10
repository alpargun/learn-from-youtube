
#%% Imports

import csv
import datetime
from pytube import Channel, Playlist, YouTube


#%% Get channel ID from channel name

#Users with driving videos: 
#- https://www.youtube.com/@jutah/videos
#- https://www.youtube.com/@Relaxing.Scenes.Driving/videos

channel_name = "jutah" # "Relaxing.Scenes.Driving"
c = Channel(f"https://www.youtube.com/@{channel_name}/videos") # using the new '@user' requires modifying pytube source code
#c = Channel("https://www.youtube.com/channel/UCBcVQr-07MH-p9e2kRTdB3A/videos") # user_id can be used easily without modifying pytube

print(f'Channel name: {c.channel_name}')
print(f'Channel id: {c.channel_id}')

# %% Get URLs of all videos of a channel

modified_id = c.channel_id[:1] + 'U' + c.channel_id[2:] # change 1st character, C, to U for channel ID  
p = Playlist(f"https://www.youtube.com/playlist?list={modified_id}")

print("Total number of videos in the channel: ", len(p))
p

# %% Save urls in a csv file

CSV_PATH = "video-links/driving/" + str(c.channel_name) + ".csv" 

with open(CSV_PATH, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "title", "duration"])
    for url in p:
        yt = YouTube(url)

        if 'driving' in yt.title.lower() or 'drive' in yt.title.lower():

            duration = str(datetime.timedelta(seconds=yt.length))
            writer.writerow([yt.video_id, yt.title, duration])


# %%
