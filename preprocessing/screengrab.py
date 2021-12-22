from ..utils import get_sheet

import logging
import subprocess

def take_screen(df, mediapipe=True, redo=False):
  for index, vid in df.iterrows():
    if mediapipe:
      p_in = 'videos/' + vid.Folder + vid.File + '_POSE.mp4'
    else:
      p_in = 'videos/' + vid.Folder + vid.File

    if not os.path.isfile(f'{p_in}_SCREEN.jpg'):
      if vid.time_screenshot != '' and not redo:
        command = ['ffmpeg', '-ss', vid.time_screenshot, '-i', p_in, '-vframes', '1', '-q:v', '2', f'{p_in}_SCREEN.jpg']
      else:
        command = ['ffmpeg', '-sseof', '-2', '-i', p_in, '-update', '1', '-q:v', '2', f'{p_in}_SCREEN.jpg']

    logging.debug(subprocess.run(command))