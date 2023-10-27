from torch.utils.data import Dataset, DataLoader
import shutil
import cv2
from skimage import color
from models.model.single_camera_bev import *
from utilities.coord_util import ego2image_orig
from utilities.util_val.val_offical import LaneEval
from models.util.post_process import bev_instance2points_with_offset_z
from models.util.cluster import embedding_post
from models.util.load_model import load_model
from utilities.config_util import load_config_module
from tqdm import tqdm
import json
import os
import numpy as np
import copy
import time
import sys
sys.path.append('./bev_lane_det/')
gpu_id = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_id])

model_path = './bev_lane_det/training/res34_flip/latest.pth' #model path of verification

''' parameter from config '''
config_file = './bev_lane_det/tools/stellantis_dataset_config.py'
# config_file = './stellantis_gt_config.py'
configs = load_config_module(config_file)

x_range = configs.x_range
y_range = configs.y_range
meter_per_pixel = configs.meter_per_pixel


'''Post-processing parameters '''
post_conf = 0.0  # Minimum confidence on the segmentation map for clustering
post_emb_margin = 6.0  # embeding margin of different clusters
post_min_cluster_size = 10  # The minimum number of points in a cluster

# tmp path for save intermediate result
tmp_save_path = './bev_lane_det/tmp_stellantis_jointlab_results'

colors = np.array([[0, 0, 0],
                   [255, 255, 255],
                   [255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255],
                   [0, 255, 255],
                   [255, 0, 255],
                   [255, 255, 0],
                   [128, 128, 128],
                   [128, 0, 0],
                   [0, 128, 0],
                   [0, 0, 128],
                   [0, 128, 128],
                   [128, 0, 128],
                   [128, 128, 0]] * 10)


if __name__ == '__main__':
    # def val():
    model = configs.model()
    model = load_model(model,
                       model_path)
    print(model_path)
    model.cuda()
    model.eval()
    val_dataset = configs.val_dataset()
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            num_workers=8,
                            shuffle=False)
    ''' Make temporary storage files according to time '''
    time1 = int(time.time())
  #  time1 = 1689186548
    np_save_path = os.path.join(tmp_save_path, str(time1) + '_np')
    os.makedirs(np_save_path, exist_ok=True)
    viz_save_path = os.path.join(tmp_save_path, str(time1) + '_viz')
    os.makedirs(viz_save_path, exist_ok=True)
    res_save_path = os.path.join(tmp_save_path, str(time1) + '_res')
    os.makedirs(res_save_path, exist_ok=True)
    ''' get model result and save'''
    for i, item in enumerate(tqdm(val_loader)):
        if i<5000:
            continue
        image, img_path, cam_extrinsics, cam_intrinsic = item
        image = image.cuda()
        with torch.no_grad():
            pred_, pred2d_ = model(image)
            seg = pred_[0].detach().cpu()
            embedding = pred_[1].detach().cpu()
            offset_y = torch.sigmoid(pred_[2]).detach().cpu()
            z_pred = pred_[3].detach().cpu()
          #  c_pred = pred_[4].detach().cpu()
            seg2d = pred2d_[0].detach().cpu()
            embedding2d = pred2d_[1].detach().cpu()
            for idx in range(seg.shape[0]):
                bnname = 'image_{}_{}.png'.format(i, idx)
                ms, me, moffset, z = seg[idx].unsqueeze(0).numpy(), embedding[idx].unsqueeze(0).numpy(
                ), offset_y[idx].unsqueeze(0).numpy(), z_pred[idx].unsqueeze(0).numpy()#, c_pred[idx].unsqueeze(0).numpy()

                # ms, me, moffset, z = seg[idx].unsqueeze(0).numpy(), embedding[idx].unsqueeze(0).numpy(), offset_y[idx].unsqueeze(0).numpy(), z_pred[idx].unsqueeze(0).numpy()
                tmp_res_for_save = np.concatenate((ms, me, moffset, z), axis=1)
                save_path = os.path.join(np_save_path, bnname[:-4] + '.np')
              #  np.save(save_path, tmp_res_for_save)
                ''' get postprocess result and save '''
                prediction = (tmp_res_for_save[:, 0:1, :, :],
                              tmp_res_for_save[:, 1:3, :, :])
                off_y = tmp_res_for_save[:, 3:4, :, :][0][0]
                z = tmp_res_for_save[:, 4:5, :, :][0][0]
              

                canvas, ids = embedding_post(prediction, conf=post_conf, emb_margin=post_emb_margin,
                                             min_cluster_size=post_min_cluster_size, canvas_color=False)
                seg_img = colors[canvas]
                save_path = os.path.join(viz_save_path, bnname[:-4] + '_bev.png')
                cv2.imwrite(save_path, seg_img)
               

                lines = bev_instance2points_with_offset_z(canvas, max_x=x_range[1],
                                                           meter_per_pixal=(
                                                               meter_per_pixel, meter_per_pixel),
                                                               offset_y=off_y, Z=z)
               
                ms2d, me2d = seg2d[idx].unsqueeze(0).numpy(), embedding2d[idx].unsqueeze(0).numpy()
                tmp_res_for_save = np.concatenate((ms2d, me2d), axis=1)

                ''' get postprocess result and save '''
                prediction2d = (tmp_res_for_save[:, 0:1, :, :], tmp_res_for_save[:, 1:3, :, :])
                canvas2d, ids2d = embedding_post(prediction2d, conf=post_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)
                seg_img2d = colors[canvas2d]
                save_path = os.path.join(viz_save_path, bnname[:-4] + '_2d.png')

                cv2.imwrite(save_path, seg_img2d)

                img = cv2.imread(img_path[idx])

                lanes_pred = []
                index = 1
                for lane in lines:

                    pred_in_persformer = np.array([lane[0], lane[1], lane[2]])
                    lanes_pred.append(pred_in_persformer.T)
                    color = tuple(colors[index].tolist())

                    uv2 = ego2image_orig(np.array(
                                 [lane[0], lane[1], lane[2]]), cam_intrinsic[idx].numpy(), cam_extrinsics[idx].numpy())
                    img = cv2.polylines(img, [uv2[0:2, :].T.astype(int)], False, color, 6)
                    index += 1
                save_path = os.path.join(viz_save_path, bnname)
                cv2.imwrite(save_path, img)
               

                
    
    

# if __name__ == '__main__':
#     val()
