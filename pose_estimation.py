from .utils import get_sheet, landmark_to_dict, crop_image, coord

import json
import logging

import cv2
import mediapipe as mp

def estimate_pose(df, output_video=True):
  mp_pose = mp.solutions.pose
  mp_drawing = mp.solutions.drawing_utils 
  mp_drawing_styles = mp.solutions.drawing_styles

  for index, vid in df.iterrows():
    cap = cv2.VideoCapture('/content/drive/MyDrive/ML_boulder/videos/' + vid.Folder + vid.File)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    top, bottom, left, right = coord(frame_height, frame_width, vid.crop_y_side, vid.crop_y, vid.crop_x_side, vid.crop_x)
    width = right - left
    height = bottom - top
    
    data=[]

    if output_video:
      out = cv2.VideoWriter('/content/drive/MyDrive/ML_boulder/videos/' + vid.Folder + vid.File + '_POSE.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (width, height))

    with mp_pose.Pose() as pose:
      while True:
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        success, image = cap.read()

        if not success:
          break

        # Crop the image
        image = crop_image(image, left, right, bottom, top)

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        #Change here to add empty dict if no detection ?
        if not results.pose_landmarks:
          continue

        landmark = landmark_to_dict(results)
        data.append(landmark)
        
        if output_video:
          annotated_image = image.copy()
          mp_drawing.draw_landmarks(
              annotated_image,
              results.pose_landmarks,
              mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
          
          out.write(annotated_image)

    with open('/content/drive/MyDrive/ML_boulder/videos/' + vid.Folder + vid.File + '_POSE.json', 'w') as write_file:
      json.dump(data, write_file)

    if output_video:
      out.release()

    cap.release()

    logging.info(vid.File)