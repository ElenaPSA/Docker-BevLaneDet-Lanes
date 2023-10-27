import copy
import json
import os
import glob
import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from utilities.coord_util import ego2image, IPM2ego_matrix
from utilities.standard_camera_cpu import Standard_camera

import pdb


def get_intrinsic_matrix(path):
    s = cv2.FileStorage()
    s.open(path, cv2.FileStorage_READ)
    K = s.getNode('camera_matrix').mat()
    D = s.getNode('distortion_coefficients').mat()
    return K, D


def get_RT_matrix(path, cam_x=1.5, cam_y=0.0, cam_z=1.7):
    s = cv2.FileStorage()
    s.open(path, cv2.FileStorage_READ)
    RT = np.zeros((4,4), dtype=np.float64)
    RT[:3,:] = s.getNode('transformation_matrix').mat()
    RT[-1,-1] = 1.

   
    R=RT[:3,:3]
    T=RT[:3,3]
    project_g2c = np.zeros((4, 4))
    
    T1=np.zeros((3,1))
    T1[0]=-cam_x
    T1[1]=-cam_y
    T1[2]=-cam_z
    T2=R@T1 + np.expand_dims(T, axis=1)   
    
    project_g2c[:3, :3] = R
    project_g2c[:3, 3] = T2[:,0]
    project_g2c[3, 3] = 1.0

    return project_g2c



class Stellantis_dataset_with_offset_val(Dataset):
    def __init__(self, image_paths,
                 intrinsics_path, # path to camera calib
                 extrinsics_path,
                 data_trans,
                #  intrinsics_tf,
                 virtual_camera_config,
                 img_ext='png'):
        self.image_paths = image_paths
        self.img_ext = img_ext
        # self.gt_paths = gt_paths

        # ''' get all list '''
        # self.cnt_list = []
        # card_list = os.listdir(self.gt_paths)
        # for card in card_list:
        #     gt_paths = os.path.join(self.gt_paths, card)
        #     gt_list = os.listdir(gt_paths)
        #     for cnt in gt_list:
        #         self.cnt_list.append([card, cnt])

        assert os.path.isfile(intrinsics_path), "File {} not found".format(intrinsics_path)
        assert os.path.isfile(extrinsics_path), "File {} not found".format(extrinsics_path)
        self.intrinsics_path = intrinsics_path
        self.extrinsics_path = extrinsics_path

        # Load calib
        self.cam_intrinsic, D = get_intrinsic_matrix(intrinsics_path)
        self.cam_extrinsics = get_RT_matrix(extrinsics_path)
        
        

        ''' virtual camera paramter'''
        self.use_virtual_camera = virtual_camera_config['use_virtual_camera']
        self.vc_intrinsic = virtual_camera_config['vc_intrinsic']
        self.vc_extrinsics = virtual_camera_config['vc_extrinsics']
        self.vc_image_shape = virtual_camera_config['vc_image_shape']

        ''' transform loader '''
        self.trans_image = data_trans

        '''Get image list'''
        self.cnt_list = []
        pattern=image_paths + 'CAM_FRONT_view*.' + img_ext
      
        file_names = sorted(glob.glob(image_paths + 'CAM_FRONT_view*.' + img_ext))
      
        for path in file_names:
            dir = os.path.dirname(path)
            file = os.path.basename(path)
            self.cnt_list.append([dir, file])



    def __getitem__(self, idx):
        '''get image '''
        # gt_path = os.path.join(self.gt_paths, self.cnt_list[idx][0], self.cnt_list[idx][1])
        image_path = os.path.join(self.image_paths, self.cnt_list[idx][0], self.cnt_list[idx][1])
        image = cv2.imread(image_path)
       
        if self.use_virtual_camera:
            sc = Standard_camera(self.vc_intrinsic, self.vc_extrinsics, self.vc_image_shape,
                                 self.cam_intrinsic, np.linalg.inv(self.cam_extrinsics), (image.shape[0], image.shape[1]))
            trans_matrix = sc.get_matrix(height=0)
            image = cv2.warpPerspective(image, trans_matrix, self.vc_image_shape)

        #cv2.imwrite("test.png",image)
        transformed = self.trans_image(image=image)
        image = transformed["image"]

        return image, image_path,np.linalg.inv(self.cam_extrinsics), self.cam_intrinsic

    def __len__(self):
        return len(self.cnt_list)

if __name__ == '__main__':
    ''' parameter from config '''
    from utilities.config_util import load_config_module
    config_file = './bev_lane_det/tools/stellantis_dataset_config.py'
    configs = load_config_module(config_file)
    dataset = configs.val_dataset()
    print(len(dataset))
    data = dataset[5300]
