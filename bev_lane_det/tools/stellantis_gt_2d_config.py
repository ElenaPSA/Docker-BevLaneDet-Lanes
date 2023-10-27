import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.spatial.transform import Rotation as Rotation
import numpy as np
from loader.bev_road.stellantis_gt_data_2d import Stellantis_gt_2d_dataset
from loader.bev_road import sampler

from models.model.single_camera_2d import BEV_LaneDet


def get_camera_matrix(cam_pitch_deg, cam_x, cam_z):

    yaw_pitch_roll_deg = np.array([0.0, cam_pitch_deg, 0.0])
    r1 = Rotation.from_euler('zyx', yaw_pitch_roll_deg, degrees=True)
    r1 = r1.as_matrix()

    t = np.asarray([cam_x, 0.0, cam_z])

    r = np.zeros((3, 3))
    r[0, 2] = 1
    r[1, 0] = -1
    r[2, 1] = -1

    R = r.T @ r1
    T = -R @ t.T

    proj_g2c = np.zeros((4, 4))
    proj_g2c[:3, :3] = R
    proj_g2c[:3, 3] = T
    proj_g2c[3, 3] = 1.0

    camera_K = np.array([[3987.456 / 2.0, 0., 1934.125 / 2.0],
                         [0., 3995.199 / 2.0, 938.677 / 2.0],
                         [0., 0., 1.]])

    return proj_g2c, camera_K


''' data split '''
root_path = '/mnt/data/VisionDataBases/PSA/PSA_2023_dataset/gt_anonymized'
model_save_path = "./bev_lane_det/training/res34_flip_modif_2d/"
log_path='./bev_lane_det/training/res34_flip_modif_2d/logs'

input_shape = (640,1280)
output_2d_shape = (160,320)
x_range = (8, 108)
usefocalloss = True
with_validation = True

loader_args = dict(
    batch_size=8,
    num_workers=12,
    shuffle=True,
    drop_last=True
)


balance=False
start_epoch = 0
distributed = True

''' model '''
def model():
    return BEV_LaneDet(output_2d_shape=output_2d_shape, train=True)

''' optimizer '''
epochs = 35
optimizer = AdamW
optimizer_params = dict(
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False
)
scheduler = CosineAnnealingLR


def train_dataset():
    train_trans = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.MotionBlur(p=0.2),
        A.RandomBrightnessContrast(),
        A.ColorJitter(p=0.1),
        A.Normalize(),
        ToTensorV2()
    ])
    
    train_data = Stellantis_gt_2d_dataset(root_path, x_range,True,
                                            train_trans, output_2d_shape )
    if balance:
        weights = sampler.BalancedTrainingSampler.computeweights(train_data)
        weights = torch.from_numpy(weights)
        if distributed:
            sample = sampler.DistributedBalancedTrainingSampler(weights, 42)
        else:
            sample = sampler.BalancedTrainingSampler(weights, 42)
        return train_data, sample
    
    return train_data


def val_dataset():
    trans_image = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(),
        ToTensorV2()])
    val_data = Stellantis_gt_2d_dataset(root_path, x_range,False,
                                            trans_image, output_2d_shape )
    return val_data
