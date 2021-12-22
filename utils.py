import copy
import logging

import gspread as gs
import imageio
import numpy as np
import pandas as pd

from numpy.core.numeric import count_nonzero

NUMBER_OF_SHEETS = 5

left_hand = ["LEFT_THUMB.x", "LEFT_INDEX.x", "LEFT_PINKY.x",  "LEFT_WRIST.x", "LEFT_THUMB.y", "LEFT_INDEX.y", "LEFT_PINKY.y",  "LEFT_WRIST.y", "LEFT_THUMB.v", "LEFT_INDEX.v", "LEFT_PINKY.v",  "LEFT_WRIST.v"]
right_hand = ["RIGHT_THUMB.x", "RIGHT_INDEX.x", "RIGHT_PINKY.x",  "RIGHT_WRIST.x", "RIGHT_THUMB.y", "RIGHT_INDEX.y", "RIGHT_PINKY.y",  "RIGHT_WRIST.y", "RIGHT_THUMB.v", "RIGHT_INDEX.v", "RIGHT_PINKY.v",  "RIGHT_WRIST.v"]
left_foot = ["LEFT_FOOT_INDEX.x", "LEFT_ANKLE.x", "LEFT_HEEL.x", "LEFT_FOOT_INDEX.y", "LEFT_ANKLE.y", "LEFT_HEEL.y", "LEFT_FOOT_INDEX.v", "LEFT_ANKLE.v", "LEFT_HEEL.v"]
right_foot = ["RIGHT_FOOT_INDEX.x", "RIGHT_ANKLE.x", "RIGHT_HEEL.x", "RIGHT_FOOT_INDEX.y", "RIGHT_ANKLE.y", "RIGHT_HEEL.y", "RIGHT_FOOT_INDEX.v", "RIGHT_ANKLE.v", "RIGHT_HEEL.v"]
extremities = {"left_hand" : left_hand, "right_hand": right_hand, "left_foot" : left_foot, "right_foot" : right_foot}

# Used for reading the excel sheet
def get_sheet(i, path=None):
    # With google colab
    if path is None:
        gc = gs.authorize(GoogleCredentials.get_application_default())
        worksheet = gc.open('boulder_problems').get_worksheet(i)

        # get_all_values gives a list of rows.
        rows = worksheet.get_all_values()

        # Convert to a DataFrame and render.
        df = pd.DataFrame.from_records(rows)
        df.columns = df.iloc[0] 
        df = df[1:]
    # Without google colab
    else:
        df= pd.read_excel(path, sheet_name=i)
        df.columns = df.iloc[0] 
        df = df[1:]
    return df

def run_all(func, path=None, **kwargs):
  for i in range(NUMBER_OF_SHEETS):
    df = get_sheet(i, path) 
    func(df, kwargs)

def coord(height, width, side_h, frac_h, side_w, frac_w):
  frac_h = frac_h / 100
  frac_w = frac_w / 100

  left = 0
  right = width - 1
  top = 0
  bottom = height - 1


  if(side_h == 'center'):
    top = round((1 - frac_h) * height / 2)
    bottom = height - 1 - top
  elif(side_h == 'top'):
    bottom = height - 1 - round((1 - frac_h) * height)
  elif(side_h == 'bottom'):
    top = round((1 - frac_h) * height)
  

  if(side_w == 'center'):
    left = round((1 - frac_w) * width / 2)
    right = width - 1 - left
  elif(side_w == 'left'):
    right = width - 1 - round((1 - frac_w) * width)
  elif(side_w == 'right'):
    left = round((1 - frac_w) * width)
  

  return top, bottom, left, right

def crop_image(img, left, right, bottom, top):
  return img[top:bottom, left:right]

def weighted_m(x, w):
  sum_w = w.sum()
  num = np.multiply(x,w).sum()
  return (num/sum_w)

def create_gif_arrow(img, centers):
  frames = []
  timeframe = []

  frames.append(img)

  frame_width = img.shape[1]
  frame_height = img.shape[0]

  scaled_centers = copy.deepcopy(centers)

  move_order = []

  for key, coord in scaled_centers.items():
    for elem in coord:
      elem[0][0] *= frame_width
      elem[0][1] *= frame_height
      move_order.append((key, elem[1]))

  move_order.sort(key=lambda y: y[1])

  members = ['left_hand', 'right_hand', 'left_foot', 'right_foot']
  colors = {'left_hand': (0, 204, 204), 'right_hand':(0, 204, 0), 'left_foot': (255, 102, 102), 'right_foot':(76, 0, 153)}

  img = add_legend(img, colors, members)

  size = 30
  thickness = 4
  
  img_base = img.copy()

  count = dict(zip(members, [[0,0], [0,0], [0,0], [0,0]]))

  for id, move in enumerate(move_order):
    img = img_base.copy()

    for member in members:
      start = count[member][0]
      end = count[member][1]
      draw_last_moves(img, scaled_centers[member][start:end], colors[member])

    member = move[0]

    count[member][1] += 1 #add 1 to the end
    start = count[member][0] #get start and end
    end = count[member][1]
    draw_last_moves(img, scaled_centers[member][end-1:end], colors[member])


    if (count[member][1] > 1):
      draw_arrow(img, scaled_centers[member][start:end], colors[member])
      count[member][0] = count[member][1] - 1 #

    frames.append(img)
    timeframe.append(move[1])

  return frames, timeframe

def save_gif(img, dict_centers, p_out):
  frames, timeframe = create_gif_arrow(img, dict_centers)
  timeframe_scaled = [timeframe[i] - timeframe[i-1] for i in range(1,len(timeframe))] #get number of frame btween two moves
  timeframe_scaled = [timeframe_scaled[i] if timeframe_scaled[i] < 60 else 60 for i in range(len(timeframe_scaled))]#cap the maximum duration bteween two move to two seconds
  timeframe_scaled.insert(0, 30) #add boulder image without any move, at the beginning, for 1 sec
  timeframe_scaled.append(60) #stay on last image for 2 sec
  timeframe_scaled = list(np.divide(timeframe_scaled,30)) #divide by 30 to have speed of 0.5 (original 60fps)
  logging.debug(f"Saving GIF file for {p_out}")
  with imageio.get_writer(p_out, mode="I", duration = timeframe_scaled) as writer:
      for idx, frame in enumerate(frames):
          rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          writer.append_data(rgb_frame)
          logging.debug(f"Adding frame {idx+1} to GIF for {p_out}: ")

def landmark_to_dict(results):
  dict_ = { "LEFT_FOOT_INDEX": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
                                "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].visibility},
            "LEFT_ANKLE": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                           "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].visibility},
            "LEFT_HEEL": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y,
                          "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].visibility},
            "RIGHT_FOOT_INDEX": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                                "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y,
                                 "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].visibility},
            "RIGHT_ANKLE": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                          "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility},
            "RIGHT_HEEL": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y,
                           "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].visibility},
            "LEFT_THUMB": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x,
                            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y,
                           "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].visibility},
            "LEFT_INDEX": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y,
                           "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].visibility},
            "LEFT_PINKY": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y,
                           "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].visibility},
            "LEFT_WRIST": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                           "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].visibility},
            "RIGHT_THUMB": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x,
                            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y,
                            "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].visibility},
            "RIGHT_INDEX": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y,
                          "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].visibility},
            "RIGHT_PINKY": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y,
                          "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].visibility},
            "RIGHT_WRIST": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                          "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].visibility}, 
            "NOSE": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                     "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                     "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility}, 
            "RIGHT_ELBOW": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                            "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility}, 
            "LEFT_ELBOW": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                           "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].visibility},  
            "RIGHT_KNEE": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                           "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].visibility}, 
            "LEFT_KNEE": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                          "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].visibility}, 
           "RIGHT_HIP": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,
                         "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].visibility}, 
            "LEFT_HIP": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                          "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
                         "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].visibility},
           "RIGHT_SHOULDER": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                              "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility}, 
            "LEFT_SHOULDER": {"x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                              "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                              "v": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility},   
      }
  return dict_