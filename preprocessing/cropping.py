from ..utils import get_sheet

import logging
import subprocess

def crop_time(df):
  for index, vid in df.iterrows():
    p_in = 'ml/good/' + vid.Folder + vid.File
    dir = 'videos/' + vid.Folder
    p_out = dir + vid.File

    res = subprocess.run(['mkdir', '-p', dir])

    if(vid.time_start != '-1' and vid.time_end != '-1'):
      command = ['ffmpeg', '-y', '-ss', vid.time_start, '-i', p_in,  '-to', vid.time_end, '-c', 'copy', p_out]
      res = subprocess.run(command)
    elif(vid.time_start == '-1' and vid.time_end != '-1'):
      command = ['ffmpeg', '-y', '-i', p_in,  '-to', vid.time_end, '-c', 'copy', p_out]
      res = subprocess.run(command)
    elif(vid.time_start != '-1' and vid.time_end == '-1'):
      command = ['ffmpeg', '-y', '-ss', vid.time_start, '-i', p_in, '-c', 'copy', p_out]
      res = subprocess.run(command)
    else:
      command = ['ffmpeg', '-y', '-i', p_in, '-c', 'copy', p_out]
      res = subprocess.run(command)

    logging.debug(res)