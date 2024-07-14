# Learn From YouTube Videos

This repository aims to create on-demand video datasets from YouTube, which can then be used to train video networks through self-supervised learning.

A .csv file with YouTube video IDs is enough to automatically download and preprocess all videos.
Additionally, all videos from a single channel/user can be downloaded automatically.

Uses [my PyTube fork](https://github.com/alpargun/pytube) with bug fixes and enhancements to download the videos.

## Progress so far
- Can download Youtube videos from a .csv file with URLs.
- Can clip videos as $n$ frames with $m$ step between frames to prepare input for video models.
- Can sample multiple clips from the same video.
- Implements custom video dataset class and dataloaders. Visualizes clips from batches for debugging.
- Implements video transforms such as cropping, resizing, normalizaton.
- Can run inference with a downloaded 3D-ResNet50 model on downloaded K400 test set videos.
- Can run YOLOv8 inference on prepared dataset and visualizes the detections. 
- Extracts important keywords from video description and title.

## TODO

- [x] Save youtube links
- [x] Implement video downloader
- [x] Prepare video dataset class
- [x] Preprocessing/Transform for videos (e.g. resolution, clipping)
- [x] Test inference of downloaded video networks
- [x] Download title and description from videos and extract keywords to label videos
- [x] Test inference of downloaded image networks (YOLO)
- [ ] Implement own network 
- [ ] SSL training
- [ ] Evaluate on downstream CV tasks
- [ ] Evaluate on downstream RL tasks (Atari etc.) to test interacting with the world