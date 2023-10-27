import os
import copy
import cv2
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import json
import sys
sys.path.append('/data/gvincent/bev_lane_det/')
from utilities.coord_util import *
from models.util.cluster import embedding_post
from models.util.post_process import bev_instance2points_with_offset_z

model_res_save_path = '/data/gvincent/bev_lane_det/tmp/tmp_openlane/1685533675_np'
postprocess_save_path = '/data/gvincent/bev_lane_det/tmp/tmp_openlane/1685533675_result'
gt_paths = '/dataL/openlane/lane3d_1000/validation'
img_paths = '/dataL/openlane/images/validation'
viz_save_path = '/data/gvincent/bev_lane_det/tmp/tmp_openlane/1685533675_viz'
os.makedirs(viz_save_path, exist_ok=True)
valid_data = os.listdir(model_res_save_path)

# for i in range(len(valid_data)):
#     files = valid_data[i].split('.')[0].split('__')

#     cv2.imwrite(os.path.join(viz_save_path, files[1] + '.jpg'), img)



files = ['segment-17065833287841703_2980_000_3000_000_with_camera_labels', '155362886424889300']
files = ['segment-191862526745161106_1400_000_1420_000_with_camera_labels', '155008196684854600']
files = ['segment-260994483494315994_2797_545_2817_545_with_camera_labels', '150723475443834500']
gt_path = os.path.join(gt_paths, files[0], files[1] + '.json')
img_path = os.path.join(img_paths, files[0], files[1] + '.jpg')


with open(gt_path) as f:
    gt = json.load(f)

ori_img = cv2.imread(img_path)
img = ori_img.copy()
plt.imshow(img[:,:,::-1])
plt.show()

cam_w_extrinsics = np.array(gt['extrinsic'])
maxtrix_camera2camera_w = np.array([[0, 0, 1, 0],
                                    [-1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, 0, 1]], dtype=float)
cam_extrinsics = cam_w_extrinsics @ maxtrix_camera2camera_w
cam_intrinsic = np.array(gt['intrinsic'])
lanes = gt['lane_lines']


cmap = 10*[(255,0,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (192,192,192), (128,128,128), (128,0,0), (128,128,0), (0,128,0), (128,0,128), (0,128,128), (0,0,128)]#, (0,255,0)
for idx in range(len(lanes)):
    lane1 = lanes[idx]
    # np.array(lane1['xyz']).T[np.array(lane1['visibility']) == 1.0]
    lane_camera_w = np.array(lane1['xyz']).T[np.array(lane1['visibility']) == 1.0].T
    lane_camera_w = np.vstack((lane_camera_w, np.ones((1, lane_camera_w.shape[1]))))
    lane_ego = cam_w_extrinsics @ lane_camera_w  #
    uv1 = ego2image(lane_ego[:3], cam_intrinsic, cam_extrinsics)
    # img = cv2.polylines(img, [uv1[0:2, :].T.astype(int)], False, cmap[idx], 6)
    img = cv2.polylines(img, [uv1[0:2, :].T.astype(int)], False, (0,0,255), 6)

# ground truth loaded as in the dataloader
plt.figure(figsize=(16,10))
plt.imshow(img[:,:,::-1])
plt.show()


x_range = (3, 103)
y_range = (-12, 12)
meter_per_pixel = 0.5 # grid size
post_conf = -0.7 # Minimum confidence on the segmentation map for clustering
post_emb_margin = 6.0 # embeding margin of different clusters
post_min_cluster_size = 3 # The minimum number of points in a cluster

np_path = os.path.join(model_res_save_path, files[0]+'__'+files[1] + '.np.npy')
loaded = np.load(np_path)
prediction = (loaded[:, 0:1, :, :], loaded[:, 1:3, :, :])
offset_y = loaded[:, 3:4, :, :][0][0]
z_pred = loaded[:, 4:5, :, :][0][0]


canvas, ids = embedding_post(prediction, conf=post_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)
from skimage import color
seg = color.label2rgb(canvas)
plt.imshow(seg)
plt.show()

lines = bev_instance2points_with_offset_z(canvas, max_x=x_range[1],
                                          meter_per_pixal=(meter_per_pixel, meter_per_pixel),
                                          offset_y=offset_y, Z=z_pred)

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(
        x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, #norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

plt.figure(figsize=(16,10))
plt.imshow(ori_img[:,:,::-1])
lanes_pred = []
for lane in lines:
    pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
    lanes_pred.append(pred_in_persformer.T)

    uv2 = ego2image(np.array([lane[0], lane[1], lane[2]]), cam_intrinsic, cam_extrinsics)
    img = cv2.polylines(img, [uv2[0:2, :].T.astype(int)], False, (0,255,0), 6)
    colorline(uv2[0, :].T.astype(int), uv2[1, :].T.astype(int), pred_in_persformer[2], cmap='viridis')
plt.savefig(viz_save_path+'/output_z.png')
plt.show()
# ground truth loaded as in the dataloader
plt.figure(figsize=(16,10))
plt.imshow(img[:,:,::-1])
plt.savefig(viz_save_path+'/output.png')
plt.show()

