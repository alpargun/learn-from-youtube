#%% Imports

import datetime
from pytube import YouTube


#%% Set video url and create youtube object

BASE_URL = "http://youtube.com/watch?v="
video_id =  "u3_2gAuPjxM" # "_VwmNdw5BYc" # "eR5vsN1Lq4E"
video_url = BASE_URL + video_id

yt = YouTube(video_url)

#%% Create stream object

stream = yt.streams.filter(progressive=True, file_extension='mp4', resolution="360p").first() # only_video=True results in double the duration of video

#%% Show video details
# Get video details
title = yt.title
description = yt.description

print(title)
print(description) # if called before stream, returns null
print(str(datetime.timedelta(seconds=yt.length)))

#%% Extract keywords from description

import pandas as pd
from yake import KeywordExtractor

# Create a KeywordExtractor instance
kw_extractor = KeywordExtractor()

# Extract keywords from the description
keywords_d = kw_extractor.extract_keywords(description)

# Convert to pandas df and order according to relevance score
columns = ["keyword", "relevance"]
df_description = pd.DataFrame(keywords_d, columns=columns)
df_description = df_description.sort_values(by=['relevance'], ascending=False)

# Print the extracted keywords and their scores
df_description

#%% Extract keywords from the title

keywords_t = kw_extractor.extract_keywords(title)

# Convert to pandas df and order according to relevance score
df_title = pd.DataFrame(keywords_t, columns=columns)
df_title = df_title.sort_values(by=['relevance'], ascending=False)

# Print the extracted keywords and their scores
df_title

#%% Get captions/subtitles

#yt.bypass_age_gate() # required to get the subtitles
stream = yt.streams.first() # required to get following portions to work

print(yt.captions) # list caption tracks (languages)
caption = yt.captions.get_by_language_code('en-US')

#%% JSON captions - in case YouTube changes response structure, parse either XML or JSON
json_captions = caption.json_captions
print(json_captions)

#%% XML captions
xml_captions = caption.xml_captions
print(xml_captions)

# %% Directly generate readable captions
print(caption.generate_srt_captions())
