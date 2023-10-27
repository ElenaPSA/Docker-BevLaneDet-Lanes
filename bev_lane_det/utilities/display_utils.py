import os
import cv2
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utilities.coord_util import ego2image

import pdb

class Line_projector:
    def __init__(self, 
                dataset_name,
                save_path,
                bev_xrange,
                bev_yrange,
                bev_res):
        self.save_path = save_path
        self.bev_xrange = bev_xrange
        self.bev_yrange = bev_yrange
        self.bev_res = bev_res
        self.dataset_name = dataset_name
        self.cam_representation = np.array([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]], dtype=float)
        self.R_vg = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=float)
        self.R_gc = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]], dtype=float)
                # self.category_dict = {0: 'invalid',
        #                       1: 'white-dash',
        #                       2: 'white-solid',
        #                       3: 'double-white-dash',
        #                       4: 'double-white-solid',
        #                       5: 'white-ldash-rsolid',
        #                       6: 'white-lsolid-rdash',
        #                       7: 'yellow-dash',
        #                       8: 'yellow-solid',
        #                       9: 'double-yellow-dash',
        #                       10: 'double-yellow-solid',
        #                       11: 'yellow-ldash-rsolid',
        #                       12: 'yellow-lsolid-rdash',
        #                       13: 'fishbone',
        #                       14: 'others',
        #                       20: 'roadedge'}

    def draw_on_img_category(self, img_path, pred, gt, cam_w_extrinsics, cam_intrinsics):
        """
        """
        
        cam_extrinsics_persformer = copy.deepcopy(cam_w_extrinsics)
        cam_extrinsics_persformer[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(self.R_vg), cam_extrinsics_persformer[:3, :3]),
            self.R_vg), self.R_gc)
        cam_extrinsics_persformer[0:2, 3] = 0.0
        matrix_lane2persformer = cam_extrinsics_persformer @ self.cam_representation
        matrix_ours2persformer = matrix_lane2persformer @ np.linalg.inv(cam_w_extrinsics)

        # if len(pred)>0:
        #     pdb.set_trace()
        img = cv2.imread(img_path)
        for lane in pred:
            lane = np.transpose(lane)
            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            lane_ego_persformer = matrix_ours2persformer @ lane
            lane_ego_persformer[0], lane_ego_persformer[1] = lane_ego_persformer[1], -1 * lane_ego_persformer[0]
            uv = ego2image(lane[:3], cam_intrinsics, cam_w_extrinsics)
            img = cv2.polylines(img, [uv[0:2, :].T.astype(int)], False, (0,255,255), 4)

        if len(gt)>0:
            pdb.set_trace()
        for lane in gt:
            lane = np.transpose(lane)
            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            lane_ego_persformer = matrix_ours2persformer @ lane
            lane_ego_persformer[0], lane_ego_persformer[1] = lane_ego_persformer[1], -1 * lane_ego_persformer[0]
            uv = ego2image(lane[:3], cam_intrinsics, cam_w_extrinsics)
            img = cv2.polylines(img, [uv[0:2, :].T.astype(int)], False, (0,0,255), 4)

        return img

    def draw_on_bev_category(self, pred, gt):
        
        
        # create bev map
        bev_shape = (int((self.bev_xrange[1] - self.bev_xrange[0]) / self.bev_res),
                     int((self.bev_yrange[1] - self.bev_yrange[0]) / self.bev_res),
                     3)
        bev_map = np.ones(bev_shape, dtype=np.uint8)*20


        for lane in pred:
            lane_xyz_m = np.array(lane)
            lane_xyz_m[:,1] -= self.bev_xrange[0]
            lane_xyz_m[:,0] -= self.bev_yrange[0]
            lane_xy_px = lane_xyz_m[:,:2]/np.array([self.bev_res, self.bev_res])
            lane_xy_px[:,1] = bev_shape[0]-lane_xy_px[:,1]
            lane_xy_px = lane_xy_px.astype(np.int64)
            try:
                bev_map = cv2.polylines(bev_map, [lane_xy_px], False, [0,255,0], 1)
            except:
                pdb.set_trace()

        for lane in gt:
            lane_xyz_m = np.array(lane)
            lane_xyz_m[:,1] -= self.bev_xrange[0]
            lane_xyz_m[:,0] -= self.bev_yrange[0]
            lane_xy_px = lane_xyz_m[:,:2]/np.array([self.bev_res, self.bev_res])
            lane_xy_px[:,1] = bev_shape[0]-lane_xy_px[:,1]
            lane_xy_px = lane_xy_px.astype(np.int64)
            bev_map = cv2.polylines(bev_map, [lane_xy_px],False, [0,0,255], 1)        

        return bev_map
