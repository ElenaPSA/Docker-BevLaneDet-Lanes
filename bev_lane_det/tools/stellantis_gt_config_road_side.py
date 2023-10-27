import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from scipy.spatial.transform import Rotation as Rotation
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from bev_lane_det.loader.bev_road import sampler

# from bev_lane_det.loader.bev_road.stellantis_gt_data import Stellantis_gt_dataset_with_offset
from bev_lane_det.loader.bev_road.stellantis_gt_visible_data import (
    Stellantis_gt_dataset_with_offset,
)


def get_camera_matrix(cam_pitch_deg, cam_x, cam_z):
    yaw_pitch_roll_deg = np.array([0.0, cam_pitch_deg, 0.0])
    r1 = Rotation.from_euler("zyx", yaw_pitch_roll_deg, degrees=True)
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

    camera_K = np.array(
        [
            [3987.456 / 2.0, 0.0, 1934.125 / 2.0],
            [0.0, 3995.199 / 2.0, 938.677 / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return proj_g2c, camera_K


""" data split """
root_path = "/data3/VisionDatabases/PSA_2023_dataset/gt_anonymized"
model_save_path = "./bev_lane_det/checkpoints/stl_roadside/"
log_path = "./bev_lane_det/logs/stl_roadside/"
input_shape = (576, 1024)
output_2d_shape = (144, 256)
usefocalloss = False
""" BEV range """
x_range = (8, 108)
y_range = (-20, 20)
meter_per_pixel = 0.5  # grid size
bev_shape = (
    int((x_range[1] - x_range[0]) / meter_per_pixel),
    int((y_range[1] - y_range[0]) / meter_per_pixel),
)

loader_args = dict(batch_size=32, num_workers=8, shuffle=True, drop_last=True)

""" virtual camera config """
camera_ext_virtual, camera_K_virtual = get_camera_matrix(
    2.5, 2.0, 1.20
)  # a random parameter
vc_config = {}
vc_config["use_virtual_camera"] = True
vc_config["vc_intrinsic"] = camera_K_virtual
vc_config["vc_extrinsics"] = np.linalg.inv(camera_ext_virtual)
vc_config["vc_image_shape"] = (1920, 960)

start_epoch = 0
classif = False
balance = False
distributed = True
roadside = True
withonlyroadside = True
gt_type = "lane_postprocessed"
motorway_only = False
with_validation = False
""" model """
if classif:
    from bev_lane_det.models.model.single_camera_bev_classif import BEV_LaneDet
else:
    from bev_lane_det.models.model.single_camera_bev import BEV_LaneDet


def model():
    return BEV_LaneDet(bev_shape=bev_shape, output_2d_shape=output_2d_shape, train=True)


""" optimizer """
load_optimizer = True
epochs = 50
optimizer = AdamW
optimizer_params = dict(
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False
)
scheduler = CosineAnnealingLR


def train_dataset():
    train_trans = A.Compose(
        [
            A.Resize(height=input_shape[0], width=input_shape[1]),
            A.MotionBlur(p=0.2),
            A.RandomBrightnessContrast(),
            A.ColorJitter(p=0.1),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    train_data = Stellantis_gt_dataset_with_offset(
        root_path,
        True,
        gt_type,
        x_range,
        y_range,
        meter_per_pixel,
        train_trans,
        output_2d_shape,
        vc_config,
        roadside,
        withonlyroadside,
    )
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
    trans_image = A.Compose(
        [
            A.Resize(height=input_shape[0], width=input_shape[1]),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    val_data = Stellantis_gt_dataset_with_offset(
        root_path,
        False,
        gt_type,
        x_range,
        y_range,
        meter_per_pixel,
        trans_image,
        output_2d_shape,
        vc_config,
        roadside,
        withonlyroadside,
    )
    return val_data
