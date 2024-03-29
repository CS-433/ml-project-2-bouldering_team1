# Move sequence detection on bouldering problems

The goal of this project was to detect the move sequence of a boulder using bouldering videos.

We were provided a dataset of videos. In this dataset, we first selected good videos and stabilised them. We then labeled all the usable videos for further preproccessing like cropping, cutting or taking a screenshot for visualization. Afterward, we ran a pose estimation algorithm in order to get the coordinates of the body parts of the climbers. We used this data to detect the moves of the boulder problem and we found the sequence using a clustering algorithm. Finally, we implemented some visualization functions to show our results.

We were pretty impressed with our results, when the climber is fully in the frame, the program outputs the right move sequence. 

Link to more videos to test the program: https://drive.google.com/drive/folders/1S0hvjk2Zq7UDENHD-uOba7GCrgQQlS4s?usp=sharing

## Prerequisites

To install the following dependencies, you can run `pip3 install -r requirements.txt` : 

* `python=~3.8` 
* `numpy`
* `pandas`
* `ffmpeg` with `vidstab` to stabilize the videos.
* `mediapipe` to estimate the climbers' poses.
* `imageio` for generating the GIF visualization.
* `gspread` for the google drive integration
* `opencv-python` for the visualization.
* `openpyxl` for the xlsx import.
* `scikit-learn` to compute the clusters for the holds.

## File structure

The file structure to follow for the project to work out of the box is the following :
```
ml-project-2-bouldering_team1
│   boulder_problems.xlsx
|   colab_notebook.ipynb
|   move_sequence.py
|   pose_estimation.py    
│   README.md
|   run.py    
|   utils.py    
│
└───preprocessing
│   │   cropping.py
│   │   screengrab.py
│   │   stabilization.py
│   
└───videos
    |
    |
    └───boulder_1_01
    |    |
    |    |
    |    └───fail
    |    |   |   [vid_file]
    |    |   |   [vid_file]
    |    |   |   ...
    |    |
    |    |
    |    └───success
    |        |   [vid_file]
    |        |   [vid_file]
    |        |   ...
    |
    └───boulder_1_02
    |    |
    |    |
    |    └───fail
    |    |   |   [vid_file]
    |    |   |   [vid_file]
    |    |   |   ...
    |    |
    |    |
    |    └───success
    |        |   [vid_file]
    |        |   [vid_file]
    |        |   ...
    └───...
```

## What it does

### Preprocessing

* Stabilization is done using FFMPEG in conjunction with vidstab, we first compute the transformation of the camera with the following command :  
`ffmpeg -hide_banner -loglevel error -i f'{[vid_file]}' -vf f'vidstabdetect=result={[vid_file]}.trf' -f null -`   
Afterwards we can stabilize the video with :   
`ffmpeg -hide_banner -loglevel error -i f'{[vid_file]}' -vf f'vidstabtransform=input={[vid_file]}.trf:smoothing=0' f'{[vid_file]}_STAB.MOV'`

* Cropping in time, we have to define the time interval we want to keep for each video in the excel sheet.

* Cropping in space, we have to define part of the video we want to keep for each video in the excel sheet.

* Screenshots, these will help us with the visualization later on. The time to take the screenshot can be defined in the excel sheet, otherwise it will be the last frame of the video.

### Pose estimation

We used mediapipe for pose estimation as it works well and ran pretty fast during our testing.

### Move sequence

For the computing the move sequence, we first check which extremity has not moved for more than a certain threshold during a short period of time, this allows us to roughly determine the coordinates of the holds. After that we run a clustering algorithm on this point cloud in order to get the centroids and to get a better idea of the holds' locations (here, by holds, we mean where the climber is either holding a hold or taking support on the wall). 

## Usage

* `--table_path` the path of the excel file, default is `boulder_problems`.
* `--n_sheet` number of sheets to process in the excel file, default is 1.
* `--path` the path of the climbing videos (the top folder, in our structure it would be `videos/`).
* `--n_boulders`, with folders of the format `boulder_i_*` in `videos/`, set the max i to explore for the stabilization, default is 7.
* `--vid_list` optional list of videos to stabilize, default is `None`.
* `--no_prep` skips all the preprocessing steps (stabilization, cropping and screengrabing).
* `--stab` to stabilize all the unstabilized videos.
* `--crop` to crop the time of the videos (according to the numbers in the excel sheet).
* `--screen` to generate the screenshot for each of the videos.
* `--pose` to estimate the pose using mediapipe and generate json files containing the keypoints coordinates.
* `--output_video` to output the videos with the pose estimation.
* `--move` to generate the move sequence using the json files of the pose estimation.
* `--gif` to save the move sequence as a GIF.
* `--normal_screens` to grab screenshots on the non mediapipe videos, defaults to False.
* `--redo_screens` to rerun the screengrabing on all videos.
* `--redo_moves` to rerun the move sequence computations.
* `--verbose`, `-v` to set the verbose level, `-v` for infos and `-vv` for debugging.

## Output

After running everything, for each video we will end up having :

* `[vid_file]`
* `[vid_file].trf`
* `[vid_file]_STAB.MOV`
* `[vid_file]_STAB.MOV_SCREEN.jpg`
* `[vid_file]_STAB.MOV_POSE.json` 
* `[vid_file]_STAB.MOV_POSE.mp4` 
* `[vid_file]_STAB.MOV_MOVE_SEQ.jpg`
* `[vid_file]_STAB.MOV_MOVE_SEQ.gif`

## Example commands

The command `python3 -m run --stab --crop --screen --pose --move` gives the following results :

![Example result](docs/res.jpg)

And the command `python3 -m run --no_prep --move --gif`, ran with a preprocessed dataset, yields the following kind of GIF :

![Example GIF result](docs/res.gif)
