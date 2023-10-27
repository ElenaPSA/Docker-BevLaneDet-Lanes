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

def liangbarsky(left, top, right, bottom, x1, y1, z1, x2, y2, z2):
    """Clips a line to a rectangular area.

    This implements the Liang-Barsky line clipping algorithm.  left,
    top, right and bottom denote the clipping area, into which the line
    defined by x1, y1 (start point) and x2, y2 (end point) will be
    clipped.

    If the line does not intersect with the rectangular clipping area,
    four None values will be returned as tuple. Otherwise a tuple of the
    clipped line points will be returned in the form (cx1, cy1, cx2, cy2).
    """
    dx = x2 - x1 * 1.0
    dy = y2 - y1 * 1.0
    if z1!=None:
        dz = z2 - z1 * 1.0
        zz1 = z1
    dt0, dt1 = 0.0, 1.0
    xx1 = x1
    yy1 = y1

    checks = ((-dx, x1 - left),
            (dx, right - x1),
            (-dy, y1 - top),
            (dy, bottom - y1))

    for p, q in checks:
        if p == 0 and q < 0:
            return None
        if p != 0:
            dt = q / (p * 1.0)
            if p < 0:
                dt0 = max(dt0, dt)
            else:
                dt1 = min(dt1, dt)
                
    if dt0>dt1:
        return None
    elif dt0<0<1<dt1:
        return None
    elif 0<dt0<dt1<1:
        x1 += dt0 * dx
        y1 += dt0 * dy
        x2 = xx1 + dt1 * dx
        y2 = yy1 + dt1 * dy
        if z1!=None:
            z1 += dt0 * dz
            z2 = zz1 + dt1 * dz
            return [[x1, y1, z1], [x2, y2, z2]]
        else:
            return [[x1, y1], [x2, y2]]
    elif 0<dt0<1:
        x1 += dt0 * dx
        y1 += dt0 * dy
        if z1!=None:
            z1 += dt0 * dz
            return [x1, y1, z1]
        else:
            return [x1, y1]
    else: # 0<dt1<1
        x2 = xx1 + dt1 * dx
        y2 = yy1 + dt1 * dy
        if z1!=None:
            z2 = zz1 + dt1 * dz
            return [x2, y2, z2]
        else:
            return [x2, y2]


class Stellantis_gt_dataset_with_offset(StellantisLaneDataset):
    def __init__(self, dataset_base_dir, is_train, gt_type, motorway_only,
                 x_range,
                 y_range,
                 meter_per_pixel,
                 data_trans,
                 output_2d_shape,
                 virtual_camera_config,
                 roadside):
        super(Stellantis_gt_dataset_with_offset, self).__init__(
            dataset_base_dir, is_train, gt_type, motorway_only=motorway_only)
        self.is_train = is_train
        self.x_range = x_range
        self.y_range = y_range
        self.meter_per_pixel = meter_per_pixel
        self.cnt_list = []
        self.lane3d_thick = 1
        self.lane2d_thick = 6
        self.dataset_base_dir = dataset_base_dir
        self.flip=False
        self.roadside = roadside
        ''' virtual camera paramter'''
        self.use_virtual_camera = virtual_camera_config['use_virtual_camera']
        self.vc_intrinsic = virtual_camera_config['vc_intrinsic']
        self.vc_extrinsics = virtual_camera_config['vc_extrinsics']
        self.vc_image_shape = virtual_camera_config['vc_image_shape']
        self.type2class = {58: 1,  # 'whole lane white'
                           60: 2,  # 'dashed lane white'
                           66: 3}  # 'roadside'

        ''' transform loader '''
        self.output2d_size = output_2d_shape
        self.trans_image = data_trans
        self.ipm_h, self.ipm_w = int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel), int(
            (self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)

    def get_y_offset_and_z(self, res_d, category_d):
        def caculate_distance(base_points, lane_points, lane_z, lane_points_set):
            condition = np.where(
                (lane_points_set[0] == int(base_points[0])) & (lane_points_set[1] == int(base_points[1])))
            if len(condition[0]) == 0:
                return None, None
            lane_points_selected = lane_points.T[condition]  # 找到bin
            lane_z_selected = lane_z.T[condition]
            offset_y = np.mean(lane_points_selected[:, 1]) - base_points[1]
            z = np.mean(lane_z_selected[:, 1])
            # @distances.argmin(),distances[min_idx] #1#lane_points_selected[distances.argmin()],distances.min()
            return offset_y, z

        # 画mask
        res_lane_points = {}
        res_lane_points_z = {}
        res_lane_points_bin = {}
        res_lane_points_set = {}
        res_lane_category = {}
        for idx in res_d:
         
            ipm_points_ = np.array(res_d[idx])
           
            ipm_points = ipm_points_.T[np.where(
                (ipm_points_[1] >= 0) & (ipm_points_[1] < self.ipm_h))].T  # 进行筛选
            if len(ipm_points[0]) <= 1:
                continue
            
            x, y, z = ipm_points[1], ipm_points[0], ipm_points[2]
            base_points = np.linspace(x.min(), x.max(),
                                      int((x.max() - x.min()) // 0.05))  # 画 offset 用得 画的非常细 一个格子里面20个点
            base_points_bin = np.linspace(int(x.min()), int(x.max()),
                                          int(int(x.max()) - int(x.min())) + 1)  # .astype(np.int)
            # print(len(x),len(y),len(y))
            if len(x) == len(set(x)):
                if len(x) <= 1:
                    continue
                elif len(x) <= 2:
                    function1 = interp1d(x, y, kind='linear',
                                         fill_value="extrapolate")  # 线性插值 #三次样条插值 kind='quadratic' linear cubic
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) <= 3:
                    function1 = interp1d(x, y, kind='quadratic',
                                         fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    function1 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='cubic')
            else:
                sorted_index = np.argsort(x)[::-1]  # 从大到小
                x_, y_, z_ = [], [], []
                for x_index in range(len(sorted_index)):  # 越来越小
                    if x[sorted_index[x_index]] >= x[sorted_index[x_index - 1]] and x_index != 0:
                        continue
                    else:
                        x_.append(x[sorted_index[x_index]])
                        y_.append(y[sorted_index[x_index]])
                        z_.append(z[sorted_index[x_index]])
                x, y, z = np.array(x_), np.array(y_), np.array(z_)
                if len(x) <= 1:
                    continue
                elif len(x) <= 2:
                    function1 = interp1d(x, y, kind='linear',
                                         fill_value="extrapolate")  # 线性插值 #三次样条插值 kind='quadratic' linear cubic
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) <= 3:
                    function1 = interp1d(x, y, kind='quadratic',
                                         fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    function1 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='cubic')

            y_points = function1(base_points)
            y_points_bin = function1(base_points_bin)
            z_points = function2(base_points)
            # cv2.polylines(instance_seg, [ipm_points.T.astype(np.int)], False, idx+1, 1)
            res_lane_points[idx] = np.array([base_points, y_points])  #
            res_lane_points_z[idx] = np.array([base_points, z_points])
            res_lane_points_bin[idx] = np.array(
                [base_points_bin, y_points_bin]).astype(np.int32)  # 画bin用的
            res_lane_points_set[idx] = np.array([base_points, y_points]).astype(
                np.int32)
            res_lane_category[idx] = category_d[idx]
        offset_map = np.zeros((self.ipm_h, self.ipm_w))
        z_map = np.zeros((self.ipm_h, self.ipm_w))
        ipm_image = np.zeros((self.ipm_h, self.ipm_w))
        category_map = np.zeros((self.ipm_h, self.ipm_w))
        for idx in res_lane_points_bin:
           
            lane_bin = res_lane_points_bin[idx].T
            category = res_lane_category[idx]
            for point in lane_bin:
                row, col = point[0], point[1]
                if not (0 < row < self.ipm_h and 0 < col < self.ipm_w):  # 没有在视野内部的去除掉
                    continue
                ipm_image[row, col] = idx
                center = np.array([row, col])
                offset_y, z = caculate_distance(center, res_lane_points[idx], res_lane_points_z[idx],
                                                res_lane_points_set[idx])  # 根据距离选idex
                if offset_y is None:
                    ipm_image[row, col] = 0
                    continue
                if offset_y > 1:
                    print('haha')
                    offset_y = 1
                if offset_y < 0:
                    print('hahahahahha')
                    offset_y = 0
                offset_map[row][col] = offset_y
                z_map[row][col] = z
                category_map[row][col] = category

        return ipm_image, offset_map, z_map,category_map
    

    def is_inside_bev(self, point):
        return self.x_range[0]<point[0]<self.x_range[1] and self.y_range[0]<point[1]<self.y_range[1]
    
    def get_bev_interpoint(self, out_point, in_point):
        top = self.y_range[0]+0.001
        left = self.x_range[0]+0.001
        bottom = self.y_range[1]-0.001
        right = self.x_range[1]-0.001
        x1 = out_point[0]
        y1 = out_point[1]
        z1 = out_point[2]
        x2 = in_point[0]
        y2 = in_point[1]
        z2 = in_point[2]
        return liangbarsky(left, top, right, bottom, x1, y1, z1, x2, y2, z2)
    
    def is_inside_img(self, point, image_shape):
        return 0<=point[0]<(image_shape[1]-1) and 0<=point[1]<(image_shape[0]-1)

    def get_img_interpoint(self, out_point, in_point, image_shape):
        left = 0
        top = 0
        right = image_shape[1]-1
        bottom = image_shape[0]-1
        x1 = out_point[0]
        y1 = out_point[1]
        z1 = out_point[2]
        x2 = in_point[0]
        y2 = in_point[1]
        z2 = in_point[2]
        return liangbarsky(left, top, right, bottom, x1, y1, z1, x2, y2, z2)

    def get_seg_offset(self, idx):

        image, gt = self.load_image_and_gt(idx)
       # print('begin', idx)
        # calculate camera parameter
        camera_k = np.asarray(gt['camera']['K'])
        R = np.asarray(gt['camera']['R'])
        T = np.asarray(gt['camera']['T'])
        project_g2c = np.zeros((4, 4))
        project_g2c[:3, :3] = R
        project_g2c[:3, 3] = T
        project_g2c[3, 3] = 1.0

        project_c2g = np.linalg.inv(project_g2c)

        # calculate point
        lane_grounds = gt['geometries']
        image_gt = np.zeros(image.shape[:2], dtype=np.uint8)
        matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel),
                        int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel)
        res_points_d = {}
        category_d = {}
        for lane_idx in range(len(lane_grounds)):
            lane_ground = np.squeeze(
                np.array(gt['geometries'][lane_idx]['body_coordinates']))
            type_id = int(gt['geometries'][lane_idx]['type_id'])
            if type_id == 997 or type_id == 998:  # ego traj
                continue
            elif (type_id == 66) and (not self.roadside):
                continue
            index = int(gt['geometries'][lane_idx]['index'])
           
            if self.flip:
                lane_ground[:,1]=-lane_ground[:,1]

            lane_ground_postprocess = []
            for i in range(lane_ground.shape[0]):
                if self.is_inside_bev(lane_ground[i]):
                    lane_ground_postprocess.append(lane_ground[i])
                else:
                    if i<lane_ground.shape[0]-1 and self.is_inside_bev(lane_ground[i+1]):
                        interpoint = self.get_bev_interpoint(lane_ground[i], lane_ground[i+1])
                        lane_ground_postprocess.append(interpoint)
                    elif i>0 and self.is_inside_bev(lane_ground[i-1]):
                        interpoint = self.get_bev_interpoint(lane_ground[i], lane_ground[i-1])
                        lane_ground_postprocess.append(interpoint)
                    # TODO: elif both outside but cross the bev area
            
            if len(lane_ground_postprocess)<2:
                continue
            lane_ground = np.asarray(lane_ground_postprocess)
            
            lane_camera = lane_ground @ R.T + T.reshape(3,)
            lane_image = (lane_camera @ camera_k.T)
            # mask = lane_image[:, 2] > 0.
            # lane_image = lane_image[mask]
            z = lane_image[:, 2]
            lane_image[:, 0] = lane_image[:, 0] / z[:]
            lane_image[:, 1] = lane_image[:, 1] / z[:]
            # lane_uv = lane_image[:, :2]
            lane_image_postprocess = []
            for i in range(lane_image.shape[0]):
                if self.is_inside_img(lane_image[i], image.shape):
                    lane_image_postprocess.append(lane_image[i])
                else:
                    if i<lane_image.shape[0]-1 and self.is_inside_img(lane_image[i+1], image.shape):
                        interpoint = self.get_img_interpoint(lane_image[i], lane_image[i+1], image.shape)
                        lane_image_postprocess.append(interpoint)
                    elif i>0 and self.is_inside_img(lane_image[i-1], image.shape):
                        interpoint = self.get_img_interpoint(lane_image[i], lane_image[i-1], image.shape)
                        lane_image_postprocess.append(interpoint)
                    # TODO: elif both outside but cross the img area

            if len(lane_image_postprocess)<2:
                continue
            lane_image = np.asarray(lane_image_postprocess)

            lane_uv = lane_image[:, :2]
          
            cv2.polylines(image_gt, [lane_uv.astype(np.int32)], False, index , self.lane2d_thick)
            x, y, z = lane_ground[:, 0], lane_ground[:, 1], lane_ground[:, 2]
            ground_points = np.array([x, y])
            ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (
                ground_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
            ipm_points_ = np.zeros_like(ipm_points)
            ipm_points_[0] = ipm_points[1]
            ipm_points_[1] = ipm_points[0]
            res_points = np.concatenate([ipm_points_, np.array([z])], axis=0)

            if index in res_points_d:
                res_points_d[index] = np.hstack((res_points_d[index], res_points))
            else:
                res_points_d[index] = res_points
           
            if index not in category_d:
                category_d[index] = self.type2class[type_id]
       
        bev_gt, offset_y_map, z_map,category_map = self.get_y_offset_and_z(res_points_d,category_d)
      #  print('finish', idx)
        ''' virtual camera '''
        if self.use_virtual_camera:
            sc = Standard_camera(self.vc_intrinsic, self.vc_extrinsics, (self.vc_image_shape[1], self.vc_image_shape[0]),
                                 camera_k, project_c2g, image.shape[:2])
            trans_matrix = sc.get_matrix(height=0)
            image = cv2.warpPerspective(image, trans_matrix, self.vc_image_shape)
           # cv2.imwrite("test_image_gt_ori.png",image_gt*10)
            image_gt = cv2.warpPerspective(image_gt, trans_matrix, self.vc_image_shape, flags=cv2.INTER_NEAREST)

        return image, image_gt, bev_gt, offset_y_map, z_map, project_c2g, camera_k, category_map


    def get_masked_lanes(self, idx):
        image, gt = self.load_image_and_gt(idx)
        # calculate camera parameter
        camera_k = np.asarray(gt['camera']['K'])
        R = np.asarray(gt['camera']['R'])
        T = np.asarray(gt['camera']['T'])

        gt_lane_grounds = gt['geometries']
        # join lane parts in a single lane
        import pandas as pd
        lane_grounds_df = pd.DataFrame(gt_lane_grounds).groupby('index').agg({'body_coordinates':'sum', 'element_id':'first', 'on_ego_side':'first', 'type_id':'first', 'type_value':'first'})
        def remove_duplicates(my_list):
            new_list = []
            for elem in my_list:
                if elem not in new_list:
                    new_list.append(elem)
            return new_list
        lane_grounds_df['body_coordinates'] = lane_grounds_df['body_coordinates'].apply(remove_duplicates)
        gt_lane_grounds = lane_grounds_df.reset_index().to_dict('records')
        gt['geometries'] = gt_lane_grounds

        # calculate intersection points
        list_grounds = []
        list_uvs = []
        list_idxs = []
        list_g2i_idxs = []
        for lane_idx in range(len(gt_lane_grounds)):
            lane_ground = np.squeeze(
                np.array(gt_lane_grounds[lane_idx]['body_coordinates']))
            type_id = int(gt_lane_grounds[lane_idx]['type_id'])
            if type_id == 997 or type_id == 998:  # ego traj
                continue
            # elif (type_id == 66) and (not self.roadside):
            #     continue
            index = int(gt_lane_grounds[lane_idx]['index'])
           
            lane_ground_postprocess = []
            for i in range(lane_ground.shape[0]):
                if self.is_inside_bev(lane_ground[i]):
                    lane_ground_postprocess.append(lane_ground[i])
                else:
                    if i<lane_ground.shape[0]-1 and self.is_inside_bev(lane_ground[i+1]):
                        interpoint = self.get_bev_interpoint(lane_ground[i], lane_ground[i+1])
                        lane_ground_postprocess.append(interpoint)
                    elif i>0 and self.is_inside_bev(lane_ground[i-1]):
                        interpoint = self.get_bev_interpoint(lane_ground[i], lane_ground[i-1])
                        lane_ground_postprocess.append(interpoint)
                    # TODO: elif both outside but cross the bev area
            
            if len(lane_ground_postprocess)<2:
                continue
            lane_ground = np.asarray(lane_ground_postprocess)
            
            lane_camera = lane_ground @ R.T + T.reshape(3,)
            lane_image = (lane_camera @ camera_k.T)
            # mask = lane_image[:, 2] > 0.
            # lane_image = lane_image[mask]
            z = lane_image[:, 2]
            lane_image[:, 0] = lane_image[:, 0] / z[:]
            lane_image[:, 1] = lane_image[:, 1] / z[:]
            # lane_uv = lane_image[:, :2]
            lane_image_postprocess = []
            ground2image_idx = []
            for i in range(lane_image.shape[0]):
                if self.is_inside_img(lane_image[i], image.shape):
                    lane_image_postprocess.append(lane_image[i])
                    ground2image_idx.append(i)
                else:
                    if i<lane_image.shape[0]-1 and self.is_inside_img(lane_image[i+1], image.shape):
                        interpoint = self.get_img_interpoint(lane_image[i], lane_image[i+1], image.shape)
                        lane_image_postprocess.append(interpoint)
                        ground2image_idx.append(-i)
                    elif i>0 and self.is_inside_img(lane_image[i-1], image.shape):
                        interpoint = self.get_img_interpoint(lane_image[i], lane_image[i-1], image.shape)
                        lane_image_postprocess.append(interpoint)
                        ground2image_idx.append(-i)
                    # TODO: elif both outside but cross the img area

            if len(lane_image_postprocess)<2:
                continue
            lane_image = np.asarray(lane_image_postprocess)

            lane_uv = lane_image[:, :2]
            list_grounds.append(lane_ground)
            list_uvs.append(lane_uv)
            list_g2i_idxs.append(ground2image_idx)
            list_idxs.append(index)
          
        return list_grounds, list_uvs, list_g2i_idxs, list_idxs, gt

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        self.flip=self.is_train and random.random() > 0.5
        image, image_gt, bev_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic, category_map = self.get_seg_offset(idx)
        #cv2.imwrite("test_image_bev_gt.png",bev_gt*20)
        #cv2.imwrite("test_image_gt.png",image_gt*20)
        T = np.array([[-1, 0, 2*self.vc_intrinsic[0,2]],
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
        

        bev_gt_instance = torch.tensor(bev_gt).unsqueeze(0)  # h, w, c0
        bev_gt_offset = torch.tensor(offset_y_map).unsqueeze(0)
        bev_gt_z = torch.tensor(z_map).unsqueeze(0)
        bev_gt_segment = torch.clone(bev_gt_instance)
        bev_gt_segment[bev_gt_segment > 0] = 1
        bev_gt_category = torch.tensor(category_map)
       
        if self.is_train:
            return image, bev_gt_segment.float(), bev_gt_instance.float(),bev_gt_offset.float(),bev_gt_z.float(),bev_gt_category.long(),image_gt_segment.float(),image_gt_instance.float()
        else:
            img_name=self.gt_seq_frameids[idx]['img_path']
            return image, bev_gt_segment.float(), bev_gt_instance.float(),bev_gt_offset.float(),bev_gt_z.float(),bev_gt_category.long(),image_gt_segment.float(),image_gt_instance.float(),img_name,cam_extrinsics, cam_intrinsic


if __name__ == '__main__':
    ''' parameter from config '''
    from utilities.config_util import load_config_module
    config_file = './bev_lane_det/tools/stellantis_gt_config.py'
    configs = load_config_module(config_file)
    dataset = configs.train_dataset()
    print(len(dataset))
    data = dataset[55]
   # print(data)
