from utils import weighted_m, save_gif, extremities

import json
import logging
import os

from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN #, KMeans
# from sklearn.metrics import silhouette_score

def check_hold_helper(nb_static_frame, index, df, threshold):
  df_frame = df.iloc[index:(index + nb_static_frame), :]
  nb_ldmk = int(df_frame.shape[1]/3)
  mean_x = np.empty([nb_static_frame, 1 ])
  mean_y = np.empty([nb_static_frame, 1 ])
  for i in range(nb_static_frame):
    visi = np.array(df_frame.iloc[i,2*nb_ldmk:])
    coord_x = np.array(df_frame.iloc[i,:(nb_ldmk)])
    coord_y = np.array(df_frame.iloc[i,(nb_ldmk):(2*nb_ldmk)])
    mean_x[i] = weighted_m(coord_x, visi)
    mean_y[i] = weighted_m(coord_y, visi)
  x = float(mean_x[0])
  y = float(mean_y[0])
  mean_x = np.abs(mean_x - mean_x[0])
  mean_y = np.abs(mean_y - mean_y[0])
  if (mean_x < threshold).all() and (mean_y < threshold).all():
    return(x, y, index)

def check_hold(body_part, data_frame):
  df_sub = data_frame[body_part]
  #hyper-parameters
  nb_static_f = 30
  threshold = 0.02
  
  res = []
  for i in range(df_sub.shape[0] - nb_static_f):
    coord = check_hold_helper(nb_static_f, i, df_sub, threshold)
    (res.append(coord) if coord is not None else None)
  return res

# def seq(df, extrem_dict):
#   result = defaultdict(list)
#   for key, part in extrem_dict.items():
#     res = check_hold(part, df)
#     res_arr = np.array([[a[0], a[1]] for a in res])

#     silhouettes = []
#     if len(res_arr) > 0:
#       for k in range(2, min(len(res_arr), 12)):
#         clustering = KMeans(n_clusters=k).fit_predict(res_arr)
#         score = silhouette_score(res_arr, clustering)
#         silhouettes.append({"k" : k, "score": score})

#       silhouettes = pd.DataFrame(silhouettes)
#       n = silhouettes.k[silhouettes.score.argmax()]

#       cluster_f = KMeans(n_clusters=n).fit(res_arr)
#       centers = cluster_f.cluster_centers_

#       for id, cl in enumerate(cluster_f.labels_):
#         if cluster_f.labels_[id - 1] != cl:
#           result[key].append((centers[cl], res[id][2]))
#   return result


def seq(df, extrem_dict):
  result = defaultdict(list)
  for key, part in extrem_dict.items():
    res = check_hold(part, df) #hold or move
    res_arr = np.array([[a[0], a[1]] for a in res])

    #hyper-parameters
    eps = 0.03
    min_sample = 20
    
    #clustering
    db = DBSCAN(eps=eps, min_samples=min_sample).fit(res_arr)

    labels = db.labels_
    
    unique_labels = set(labels)
    unique_labels.discard(-1) #to remove noise as a label

    centers = []
    for i in unique_labels:
      points_of_cluster = res_arr[labels==i,:]
      centroid_of_cluster = np.mean(points_of_cluster, axis=0) 
      centers.append(list(centroid_of_cluster))
    centers = np.array(centers)
    centers

    for i in unique_labels:
      id = list(labels).index(i)
      result[key].append((centers[i], res[id][2]))

  return result

def display_seq(img, centers):
  frame_width = img.shape[1]
  frame_height = img.shape[0]
  left_hand_color = (0, 204, 204)
  right_hand_color = (0, 204, 0)
  left_foot_color = (255, 102, 102)
  right_foot_color = (76, 0, 153)

  size = 30
  thickness = 4

  center_inorder = []

  for key, coord in centers.items():
    for elem in coord:
      x = int(elem[0][0] * frame_width)
      y = int(elem[0][1] * frame_height)
      if key == 'left_foot':
        center_inorder.append((x , y, elem[1], left_foot_color))
      elif key == 'right_foot':
        center_inorder.append((x , y, elem[1], right_foot_color))
      elif key == 'left_hand':
        center_inorder.append((x , y, elem[1], left_hand_color))
      else:
        center_inorder.append((x , y, elem[1], right_hand_color))
  
  center_inorder.sort(key = lambda x: x[2])

  for id, elem in  enumerate(center_inorder):
    x_bg = elem[0] - size
    y_bg = elem[1] - size
    x_ed = elem[0] + size
    y_ed= elem[1] + size

    cv2.rectangle(img, (x_bg, y_bg), (x_ed, y_ed), elem[3], thickness)

    cv2.putText(img, "Move:{}".format(id), (x_bg, y_bg - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, elem[3], 2)

  cv2.putText(img, "Left_hand", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, left_hand_color, 3)
  cv2.putText(img, "Right_hand", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, right_hand_color, 3)
  cv2.putText(img, "Left_foot", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, left_foot_color, 3)
  cv2.putText(img, "Right_foot", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, right_foot_color, 3)


def process_sheet(sheet, args):
  redo = args.redo_moves
  gif = args.gif
  path = args.path

  for index, vid in sheet.iterrows():
    v = path + vid.Folder + vid.File
    p_in = v + '_POSE.json'
    p_out = v + '_MOVE_SEQ.jpg'

    if not os.path.isfile(p_out) or redo or gif:
      logging.debug(f"Calculating move sequence for {v}...")

      data =json.load(open(p_in))
      df = pd.json_normalize(data)
      logging.debug(f"Loaded dataframe for {v}...")
      
      img = cv2.imread(v + '_POSE.mp4_SCREEN.jpg')
      img_gif = img.copy()

      dict_centers = seq(df, extremities)
      logging.debug(f"Calculated centers for {v}...")

      display_seq(img, dict_centers)
      logging.debug(f"Displaying move sequence for {v}...")

      cv2.imwrite(p_out, img)
      logging.debug(f"Saved {p_out}.")

      if gif:
        save_gif(img_gif, dict_centers, f'{v}_MOVE_SEQ.gif')
        logging.debug(f"Saved {v}_MOVE_SEQ.gif")