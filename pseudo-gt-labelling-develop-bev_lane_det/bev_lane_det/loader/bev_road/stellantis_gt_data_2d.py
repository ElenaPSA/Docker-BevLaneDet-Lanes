import copy
import json
import os
import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from utilities.coord_util import ego2image, IPM2ego_matrix
from utilities.standard_camera_cpu import Standard_camera
from datatools.StellantisLanedataset import StellantisLaneDataset
import random

def LinePlaneIntersection(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


class Stellantis_gt_2d_dataset(StellantisLaneDataset):
    def __init__(self,
                 dataset_base_dir, 
                 x_range,
                 is_train,
                 data_trans,
                 output_2d_shape):
        super(Stellantis_gt_2d_dataset, self).__init__(
            dataset_base_dir, is_train)
        self.is_train = is_train
        self.x_range=x_range
        self.cnt_list = []
        self.lane3d_thick = 1
        self.lane2d_thick = 6
        self.dataset_base_dir = dataset_base_dir
        self.flip=False
        ''' virtual camera paramter'''
        self.type2class = {58: 0,  # 'whole lane white'
                           60: 1,  # 'dashed lane white'
                           66: 2}  # 'roadside'

        ''' transform loader '''
        self.output2d_size = output_2d_shape
        self.trans_image = data_trans
       
    def get_seg_2d(self, idx):

        image, gt = self.load_image_and_gt(idx)
       # print('begin', idx)
        # calculate camera parameter
        self.camera_k = np.asarray(gt['camera']['K'])
        R = np.asarray(gt['camera']['R'])
        T = np.asarray(gt['camera']['T'])
        project_g2c = np.zeros((4, 4))
        project_g2c[:3, :3] = R
        project_g2c[:3, 3] = T
        project_g2c[3, 3] = 1.0

        project_c2g = np.linalg.inv(project_g2c)

        planeDir_g = np.asarray([1.0, 0.0, 0.0])
        planePt_g = np.asarray([self.x_range[0] + 0.001, 0.0, 0.0])
        planePt_g_up = np.asarray([self.x_range[1] - 0.001, 0.0, 0.0])
        planeDir = np.asarray([0.0, 0.0, 1.0])
        planePt = np.asarray([0.0, 0.0, 0.01])
        # calculate point
        lane_grounds = gt['lines']
        image_gt = np.zeros(image.shape[:2], dtype=np.uint8)
        category_map_2d= np.zeros(image.shape[:2], dtype=np.uint8)
        category_map_2d[:,:]=255
        for lane_idx in range(len(lane_grounds)):
            lane_ground = np.squeeze(
                np.array(gt['lines'][lane_idx]['body_coordinates']))
            type_id = int(gt['lines'][lane_idx]['type_id'])
            if type_id == 997 or type_id == 998:  # ego traj
                continue
            index = int(gt['lines'][lane_idx]['index'])
           
            if self.flip:
                lane_ground[:,1]=-lane_ground[:,1]

            lane_ground_postprocess = []
            for i in range(lane_ground.shape[0] - 1):
                if lane_ground[i + 1][0] <= self.x_range[0] or lane_ground[i][0] >=self.x_range[1]:
                    continue
                finalpoint=lane_ground[i]

                if lane_ground[i][0] <= self.x_range[0] and lane_ground[i + 1][0] > self.x_range[0]:
                    rayPoint = lane_ground[i]
                    rayDirection = lane_ground[i + 1] - lane_ground[i]
                    interpoint = LinePlaneIntersection(
                        planeDir_g, planePt_g, rayDirection, rayPoint, epsilon=1e-6)
                    finalpoint=interpoint
                    
                lane_ground_postprocess.append(finalpoint)

                if finalpoint[0] <= self.x_range[1] and lane_ground[i + 1][0] > self.x_range[1]:
                        lane_ground_postprocess.append(lane_ground[i])
                        rayPoint = lane_ground[i]
                        rayDirection = lane_ground[i + 1] - lane_ground[i]
                        interpoint = LinePlaneIntersection(
                            planeDir_g, planePt_g_up, rayDirection, rayPoint, epsilon=1e-6)
                        lane_ground_postprocess.append(interpoint)
                        break
                else:
                    
                    if i==(lane_ground.shape[0] - 2):
                        lane_ground_postprocess.append(lane_ground[-1])
            #lane_ground_postprocess.append(lane_ground[-1])

            
            if len(lane_ground_postprocess) <2:
                continue
            lane_ground = np.asarray(lane_ground_postprocess)
            
            lane_camera = lane_ground @ R.T + T.reshape(3,)
            lane_camera_postprocess = []
            for i in range(lane_camera.shape[0] - 1):
                if lane_camera[i + 1][2] <= 0:
                    continue
                if lane_camera[i][2] <= 0 and lane_camera[i + 1][2] > 0:
                    rayPoint = lane_camera[i]
                    rayDirection = lane_camera[i + 1] - lane_camera[i]
                    interpoint = LinePlaneIntersection(
                        planeDir, planePt, rayDirection, rayPoint, epsilon=1e-6)
                    lane_camera_postprocess.append(interpoint)
                else:
                    lane_camera_postprocess.append(lane_camera[i])

            lane_camera_postprocess.append(lane_camera[-1])
            lane_camera = np.asarray(lane_camera_postprocess)
            lane_image = (lane_camera @ self.camera_k.T)

            mask = lane_image[:, 2] > 0.
            lane_image = lane_image[mask]
            z = lane_image[:, 2]
            lane_image[:, 0] = lane_image[:, 0] / z[:]
            lane_image[:, 1] = lane_image[:, 1] / z[:]

            lane_uv = lane_image[:, :2]
          
            cv2.polylines(image_gt, [lane_uv.astype(np.int32)], False, index , self.lane2d_thick)
            cv2.polylines(category_map_2d, [lane_uv.astype(np.int32)], False, self.type2class[type_id] , self.lane2d_thick)
       
        
     

        return image, image_gt, category_map_2d

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        self.flip=self.is_train and random.random() > 0.5
        image, image_gt, category_map_2d = self.get_seg_2d(idx)
        #cv2.imwrite("image.png",image)
        #cv2.imwrite("test_category_map_2d_gt.png",category_map_2d*20)
        #cv2.imwrite("test_image_gt.png",image_gt*30)
        T = np.array([[-1, 0, 2*self.camera_k[0,2]],
                         [0,1, 0]])
        if self.flip:
            image=cv2.warpAffine(image,T,(image.shape[1],image.shape[0]))

       
        transformed = self.trans_image(image=image)
        image = transformed["image"]
        ''' 2d gt'''
        
        image_gt = cv2.resize(image_gt, (self.output2d_size[1],self.output2d_size[0]), interpolation=cv2.INTER_NEAREST)

        image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # h, w, c
        image_gt_segment = torch.clone(image_gt_instance)
        image_gt_segment[image_gt_segment > 0] = 1
        ''' 3d gt'''
        category_map_2d= cv2.resize(category_map_2d, (self.output2d_size[1],self.output2d_size[0]), interpolation=cv2.INTER_NEAREST)
        category_map_2d = torch.tensor(category_map_2d)
       
        if self.is_train:
            return image,category_map_2d.long(),image_gt_segment.float(),image_gt_instance.float()
        else:
            img_name=self.get_im_path(idx)
            return image,category_map_2d.long(),image_gt_segment.float(),image_gt_instance.float(),img_name

if __name__ == '__main__':
    ''' parameter from config '''
    from utilities.config_util import load_config_module
    config_file = './bev_lane_det/tools/stellantis_gt_2d_config.py'
    configs = load_config_module(config_file)
    dataset = configs.train_dataset()
    print(len(dataset))
    data = dataset[55]

