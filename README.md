# Learn From YouTube Videos

This repository aims to create on-demand video datasets from YouTube, which can then be used to train video networks through self-supervised learning.

A .csv file with YouTube video IDs is enough to automatically download and preprocess all videos.
Additionally, all videos from a single channel/user can be downloaded automatically.

## Progress so far
- Can download Youtube videos from a .csv file with URLs
- Can clip videos as $n$ frames with $m$ step between frames to prepare input for video models
- Implements custom video dataset class and dataloaders. Visualizes clips from batches for debugging 


## TODO

- [x] Save youtube links
- [x] Implement video downloader
- [x] Prepare video dataset class
- [ ] Preprocessing/Transform for videos (e.g. resolution, clipping)
- [ ] Implement video networks
- [ ] Implement image networks
- [ ] Implement own network 
- [ ] SSL training
- [ ] Evaluate on downstream CV tasks
- [ ] Evaluate on downstream RL tasks (Atari etc.) to test interacting with the world


## CV tasks:

- [ ] object detection
- [ ] segmentation
- [ ] facial recognition
- [ ] pose estimation
- [ ] action recognition/classification
- [ ] video compression

## Video categories:
- [x] Driving
- [x] Atari games
- [ ] Animals
- [ ] Humans
