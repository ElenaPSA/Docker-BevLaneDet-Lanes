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

model_path = './bev_lane_det/training/dla34_flip_2d/best.pth' #model path of verification

''' parameter from config '''
config_file = './bev_lane_det/tools/stellantis_gt_2d_config.py'
# config_file = './stellantis_gt_config.py'
configs = load_config_module(config_file)

x_range = configs.x_range



'''Post-processing parameters '''
post_conf = 0.0  # Minimum confidence on the segmentation map for clustering
post_emb_margin = 6.0  # embeding margin of different clusters
post_min_cluster_size = 10  # The minimum number of points in a cluster

# tmp path for save intermediate result
tmp_save_path = './bev_lane_det/tmp_stellantis_2d_results'

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
                            batch_size=8,
                            num_workers=8,
                            shuffle=False)
    ''' Make temporary storage files according to time '''
    time1 = int(time.time())
 
    viz_save_path = os.path.join(tmp_save_path, str(time1) + '_viz')
    os.makedirs(viz_save_path, exist_ok=True)
   
    ''' get model result and save'''
    for i, item in enumerate(tqdm(val_loader)):
       
        image,image_category, image_gt_segment, image_gt_instance, img_name = item
        image = image.cuda()
        with torch.no_grad():
            pred_2d, emb_2d, category_2d  = model(image)
           
            seg2d = pred_2d.detach().cpu()
            embedding2d = emb_2d.detach().cpu()
            category_2d = category_2d.detach().cpu()
            for idx in range(seg2d.shape[0]):
                bnname = 'image_{}_{}.png'.format(i, idx)
                
               
                


                ms2d, me2d = seg2d[idx].unsqueeze(0).numpy(), embedding2d[idx].unsqueeze(0).numpy()
                tmp_res_for_save = np.concatenate((ms2d, me2d), axis=1)

                ''' get postprocess result and save '''
                prediction2d = (tmp_res_for_save[:, 0:1, :, :], tmp_res_for_save[:, 1:3, :, :])
                canvas2d, ids2d = embedding_post(prediction2d, conf=post_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)
                seg_img2d = colors[canvas2d]
                save_path = os.path.join(viz_save_path, bnname[:-4] + '_2d.png')

                cv2.imwrite(save_path, seg_img2d)

                img = cv2.imread(img_name[idx])

                seg_img2d=cv2.resize(seg_img2d, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = np.where( (seg_img2d!=[0,0,0]).any(axis=2))
               
              
                img[mask]=seg_img2d[mask]
                save_path = os.path.join(viz_save_path, bnname)
                cv2.imwrite(save_path, img)

                type_ = category_2d[idx].argmax(dim=0) + 1
                type_[canvas2d==0] = 0
                classes = colors[type_]
        
                save_path = os.path.join(viz_save_path, bnname[:-4] + '_class.png')
                cv2.imwrite(save_path, classes)
                
    
    
# if __name__ == '__main__':
#     val()
