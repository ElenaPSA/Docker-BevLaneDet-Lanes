import os
import copy
import cv2
import numpy as np
import json
import sys
sys.path.append('/data/gvincent/bev_lane_det/')
import matplotlib.pyplot as plt
from skimage import color
from glob import glob
from scipy.spatial import distance

from utilities.coord_util import *
from models.util.cluster import embedding_post_centers
from models.util.post_process import bev_instance2points_with_offset_z

model_res_save_path = '/data/gvincent/bev_lane_det/tmp/tmp_openlane/1684999630_np'
postprocess_save_path = '/data/gvincent/bev_lane_det/tmp/tmp_openlane/1684999630_result'
gt_paths = '/dataL/openlane/lane3d_1000/validation'
img_paths = '/dataL/openlane/images/validation'


files = ['segment-17065833287841703_2980_000_3000_000_with_camera_labels', '155362886424889300']
files = ['segment-191862526745161106_1400_000_1420_000_with_camera_labels', '155008196684854600']
files = ['segment-260994483494315994_2797_545_2817_545_with_camera_labels', '150723475443834500']
gt_path = os.path.join(gt_paths, files[0], files[1] + '.json')
img_path = os.path.join(img_paths, files[0], files[1] + '.jpg')

np_paths = sorted(glob(model_res_save_path+'/'+files[0]+'*'))
# np_path = os.path.join(model_res_save_path, files[0]+'__'+files[1] + '.np.npy')

post_conf = -0.7 # Minimum confidence on the segmentation map for clustering
post_emb_margin = 6.0 # embeding margin of different clusters
post_min_cluster_size = 3 # The minimum number of points in a cluster
gap = post_emb_margin
    
previous_centers = []
for np_path in np_paths:
    loaded = np.load(np_path)
    prediction = (loaded[:, 0:1, :, :], loaded[:, 1:3, :, :])
    offset_y = loaded[:, 3:4, :, :][0][0]
    z_pred = loaded[:, 4:5, :, :][0][0]
    canvas, ids, centers = embedding_post_centers(prediction, conf=post_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)

    matches = []
    for new_id, new_c in enumerate(centers):
        min_gap = gap + 1
        min_cid = -1
        for old_id, old_c in enumerate(previous_centers):
            diff = distance.euclidean(new_c[0], old_c[0])
            if diff < min_gap:
                min_gap = diff
                min_cid = old_id
        if min_gap < gap:
            matches.append([new_id+1, min_cid+1])
        else:
            matches.append([new_id+1, -1])
    previous_centers = centers

    print(matches)

    fig, (ax1, ax2) = plt.subplots(1,2)
    seg = color.label2rgb(canvas)
    ax1.imshow(seg)
    ax2.imshow(np.reshape(list(dict(zip(canvas.flatten(), seg.reshape(-1,3))).values()), (-1,1,3)))
    plt.show()
    # fig, (ax1, ax2) = plt.subplots(1,2)
    # ax1.imshow(loaded[0,1])
    # ax2.imshow(loaded[0,2])
    # plt.show()
    # lines = bev_instance2points_with_offset_z(canvas, max_x=103,
    #                                             meter_per_pixal=(0.5,0.5),
    #                                             offset_y=offset_y, Z=z_pred)
