import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from loader.bev_road.stellantis_data import Stellantis_dataset_with_offset_val
from models.model.single_camera_bev import BEV_LaneDet
from scipy.spatial.transform import Rotation as Rotation

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
# train_gt_paths = '/dataset/openlane/lane3d_1000/training'
# train_image_paths = '/dataset/openlane/images/training'
val_gt_paths = ''
val_image_paths = '/mnt/data/VisionDataBases/PSA/DatabaseCamLidar/20220127_161341_Rec_JLAB09/CAM_FRONT/'
intrinsics_path = '/mnt/data/VisionDataBases/PSA/DataEcon/20220127_161341_Rec_JLAB09/Image1Calib.yaml'
extrinsics_path = '/mnt/data/VisionDataBases/PSA/DataEcon/20220127_161341_Rec_JLAB09/Image1Calib.yaml'

model_save_path = "/dataset/model/openlane"

input_shape = (576,1024)
output_2d_shape = (144,256)

''' BEV range '''
x_range = (8, 108)
y_range = (-20, 20)
meter_per_pixel = 0.5 # grid size
bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel),int((y_range[1] - y_range[0]) / meter_per_pixel))

loader_args = dict(
    batch_size=64,
    num_workers=12,
    shuffle=True
)

''' virtual camera config '''
vc_config = {}
vc_config['use_virtual_camera'] = True
''' virtual camera config '''
camera_ext_virtual, camera_K_virtual = get_camera_matrix(
    2.5, 2.0, 1.20)  # a random parameter
vc_config = {}
vc_config['use_virtual_camera'] = True
vc_config['vc_intrinsic'] = camera_K_virtual
vc_config['vc_extrinsics'] = np.linalg.inv(camera_ext_virtual)
vc_config['vc_image_shape'] = (1920, 960)



''' model '''
def model():
    return BEV_LaneDet(bev_shape=bev_shape, output_2d_shape=output_2d_shape,train=True)


''' optimizer '''
epochs = 50
optimizer = AdamW
optimizer_params = dict(
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False
)
scheduler = CosineAnnealingLR



# def train_dataset():
#     train_trans = A.Compose([
#                     A.Resize(height=input_shape[0], width=input_shape[1]),
#                     A.MotionBlur(p=0.2),
#                     A.RandomBrightnessContrast(),
#                     A.ColorJitter(p=0.1),
#                     A.Normalize(),
#                     ToTensorV2()
#                     ])
#     train_data = Stellantis_dataset_with_offset(train_image_paths, train_gt_paths, 
#                                               x_range, y_range, meter_per_pixel, 
#                                               train_trans, output_2d_shape, vc_config)

#     return train_data


def val_dataset():
    trans_image = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(),
        ToTensorV2()])
    
    # intrinsics_tf = {
    #     "fx_scale": input_shape[1]/1920,
    #     "fy_scale": input_shape[0]/1080,
    #     "cx_offset": 0,
    #     "cy_offset": 0}
    
    val_data = Stellantis_dataset_with_offset_val(val_image_paths,
                                                  intrinsics_path,
                                                  extrinsics_path,
                                                  trans_image,
                                                #   intrinsics_tf,
                                                  vc_config,
                                                  'png')
    return val_data



