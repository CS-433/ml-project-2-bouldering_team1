import logging
import subprocess

def crop_time(df, args):
  path = args.path
  for index, vid in df.iterrows():
    p_in = path + vid.Folder + vid.File
    _dir = path + vid.Folder
    p_out = _dir + vid.File

    res = subprocess.run(['mkdir', '-p', _dir])

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