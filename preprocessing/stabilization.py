import logging
import os
import re
import subprocess

#Functions to find unprocessed vids
def find_stab_vids(path, n_boulders):
  res = []
  for up in os.listdir(path):
    if re.search(r'boulder_[1-' + str(n_boulders) + r']_.*$', up):
      for inside in os.listdir(f'{path}{up}/'):
        if len(inside) > 0 and os.path.isdir(f'{path}{up}/{inside}'):
          for v in os.listdir(f'{path}{up}/{inside}'):
            if re.search(r'(^.*_STAB.MOV$)', v):
              res.append(f'{path}{up}/{inside}/{v}')

  return res

def find_all_vids(path, n_boulders):
  res = []
  for up in os.listdir(path):
    if re.search(r'boulder_[1-' + str(n_boulders) + r']_.*$', up):
      for inside in os.listdir(f'{path}{up}/'):
        if len(inside) > 0 and os.path.isdir(f'{path}{up}/{inside}'):
          for v in os.listdir(f'{path}{up}/{inside}'):
            if re.search(r'(^.*.MOV$)|(^MOV_.*.mp4$)|(^.*.MP4$)', v):
              res.append(f'{path}{up}/{inside}/{v}')

  return res

def find_unstab_vids(path, n_boulders):
  all_vids = find_all_vids(path, n_boulders)
  stab_vids = find_stab_vids(path, n_boulders)
  non_stab_vids = list(set(all_vids) - set(stab_vids))
  left_vids = [v[:-9] for v in list(set([f'{v}_STAB.MOV' for v in non_stab_vids]) - set(stab_vids))]
  if len(left_vids) == 0:
    logging.info("All done, no videos to stabilize !")
  return left_vids


#Used for stabilizing all the videos in a given list of paths (by default the still unprocessed vids)
def stabilize(args):
  path = args.path
  n_boulders = args.n_boulders
  vid_list = args.vid_list

  if vid_list is None:
    vid_list = find_unstab_vids(path, n_boulders)
    
  for v in vid_list:
    transform_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', f'{v}', '-vf', f'vidstabdetect=result={v}.trf', '-f', 'null', '-']
    stab_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', f'{v}', '-vf', f'vidstabtransform=input={v}.trf:smoothing=0', f'{v}_STAB.MOV']
    
    print(f"*** Processing {v}...")
    if subprocess.run(transform_command).returncode == 0:
      if subprocess.run(stab_command).returncode == 0:
        logging.debug("All good, stabilized version saved")
      else:
        logging.warn("There was an error running the stabilisation script")
    else:
      logging.warn("There was an error running the transform script")