import json
import os
import shutil
import sys
import time
from glob import glob

import cv2
import numpy as np
import torch
from loguru import logger
from models.util.cluster import embedding_post
from models.util.load_model import load_model
from models.util.post_process import bev_instance2points_with_offset_z
from scipy import interpolate
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities.config_util import load_config_module
from utilities.coord_util import ego2image_orig
from utilities.util_val.val_offical import LaneEval

logger.remove()
logger.add(sys.stdout, level="INFO")

gpu_id = [1, 2, 3]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_id])

BATCH_SIZE = 1

# TODO: change model path
model_path = "./bev_lane_det/checkpoints/latest.pth"  # model path of verification
# model_path = "./bev_lane_det/checkpoints/stl_roadside/latest.pth"

""" parameter from config """
config_file = "./bev_lane_det/tools/stellantis_gt_config.py"
# config_file = "./bev_lane_det/tools/stellantis_gt_config_road_side.py"
configs = load_config_module(config_file)

# TODO: choose if you want to evaluate or visualize the model predictions (both can be set to True at the same time)
evaluate = True
save_viz = True

x_range = configs.x_range
y_range = configs.y_range
meter_per_pixel = configs.meter_per_pixel


"""Post-processing parameters """
post_conf = 0.0  # Minimum confidence on the segmentation map for clustering
post_emb_margin = 6.0  # embeding margin of different clusters
post_min_cluster_size = 10  # The minimum number of points in a cluster

# TODO: change save path
# tmp path for save intermediate result
tmp_save_path = "./bev_lane_det/tmp_stellantis_results"

colors = np.array(
    [
        [0, 0, 0],
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
        [128, 128, 0],
    ]
    * 10
)


def treat_one_image(
    model,
    image,
    img_name,
    cam_extrinsics,
    cam_intrinsic,
    image_for_viz,
    gt,
    classif,
    viz_save_path,
    res_save_path,
):
    """Performs inference and saves all results on one image
    TODO: add more info.
    Arguments:
    ----------
        image,
        img_name,
        cam_extrinsics,
        cam_intrinsic,
    """
    image = image.cuda()
    with torch.no_grad():
        # 1. Model inference step on pne batch
        pred_, pred2d_ = model(image)
        seg = pred_[0].detach().cpu()
        embedding = pred_[1].detach().cpu()
        offset_y = torch.sigmoid(pred_[2]).detach().cpu()
        z_pred = pred_[3].detach().cpu()
        if classif:
            c_pred = pred_[4].detach().cpu()
        seg2d = pred2d_[0].detach().cpu()
        embedding2d = pred2d_[1].detach().cpu()

        if isinstance(img_name, torch.Tensor):
            img_name = list(img_name.detach().cpu())
        if isinstance(cam_extrinsics, torch.Tensor):
            cam_extrinsics = list(cam_extrinsics.detach().cpu())
        if isinstance(cam_intrinsic, torch.Tensor):
            cam_intrinsic = list(cam_intrinsic.detach().cpu())
        if isinstance(image_for_viz, torch.Tensor):
            image_for_viz = np.array(image_for_viz.detach().cpu())

        for idx in range(seg.shape[0]):
            # TODO: Generalize the code for any image naming convention
            bname = (
                img_name[idx].split("/")[-5] + "_" + img_name[idx].split("_")[-1][:-4]
            )
            # 2. Postprocess one image (bev and 2d)
            ms, me, moffset, z = (
                seg[idx].unsqueeze(0).numpy(),
                embedding[idx].unsqueeze(0).numpy(),
                offset_y[idx].unsqueeze(0).numpy(),
                z_pred[idx].unsqueeze(0).numpy(),
            )
            if classif:
                c = c_pred[idx].unsqueeze(0).numpy()
            else:
                c = None

            tmp_res_for_save = np.concatenate((ms, me, moffset, z), axis=1)
            save_path = os.path.join(res_save_path, bname + "_raw_pred.npy")
            # np.save(save_path, tmp_res_for_save)
            dict_to_save = {
                "pred": tmp_res_for_save,
                "cam_extrinsics": cam_extrinsics[idx],
                "cam_intrinsics": cam_intrinsic[idx],
            }
            np.save(save_path, dict_to_save)

            prediction = (
                tmp_res_for_save[:, 0:1, :, :],
                tmp_res_for_save[:, 1:3, :, :],
            )
            off_y = tmp_res_for_save[:, 3:4, :, :][0][0]
            z = tmp_res_for_save[:, 4:5, :, :][0][0]

            canvas, _ = embedding_post(
                prediction,
                conf=post_conf,
                emb_margin=post_emb_margin,
                min_cluster_size=post_min_cluster_size,
                canvas_color=False,
            )

            lines = bev_instance2points_with_offset_z(
                canvas,
                max_x=x_range[1],
                meter_per_pixal=(meter_per_pixel, meter_per_pixel),
                offset_y=off_y,
                Z=z,
            )

            # This part is only needed if one saves the vis, so it can probably be moved
            # back if perfs in this case are needed
            ms2d, me2d = (
                seg2d[idx].unsqueeze(0).numpy(),
                embedding2d[idx].unsqueeze(0).numpy(),
            )
            tmp_res_for_save2d = np.concatenate((ms2d, me2d), axis=1)

            prediction2d = (
                tmp_res_for_save2d[:, 0:1, :, :],
                tmp_res_for_save2d[:, 1:3, :, :],
            )
            canvas2d, _ = embedding_post(
                prediction2d,
                conf=post_conf,
                emb_margin=post_emb_margin,
                min_cluster_size=post_min_cluster_size,
                canvas_color=False,
            )

            # 3. Prepare output for saving in the .json format
            frame_lanes_pred = []
            for lane in lines:
                pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
                frame_lanes_pred.append(pred_in_persformer.T.tolist())

            frame_lanes_gt = prepare_frame_gt(gt[idx])
            save_path = os.path.join(res_save_path, bname + "_gt_pred.json")
            with open(save_path, "w") as f1:
                json.dump([frame_lanes_pred, frame_lanes_gt], f1)

            # 4. Prepare vizualization
            if save_viz:
                save_path = os.path.join(viz_save_path, bname)
                save_bev_images(canvas, save_path, classif, c)

                seg_img2d = colors[canvas2d]

                cv2.imwrite(save_path + "_2d.png", seg_img2d)
                logger.debug(f"{img_name[idx]}, {bname}")
                img = image_for_viz[idx]
                img_inter = np.copy(img)

                def add_lane_to_2d_canvas(image, x, y, z, color):
                    uv = ego2image_orig(
                        np.array([x, y, z]),
                        cam_intrinsic[idx].numpy(),
                        cam_extrinsics[idx].numpy(),
                    )
                    image = cv2.polylines(
                        image, [uv[0:2, :].T.astype(int)], False, color, 6
                    )
                    return image

                for index, lane in enumerate(lines):
                    color = tuple(colors[index + 1].tolist())
                    # Without interpolation
                    img = add_lane_to_2d_canvas(img, lane[0], lane[1], lane[2], color)

                    # With interpolation
                    spline = interpolate.make_smoothing_spline(lane[0], lane[1], lam=10)
                    ynew = spline(lane[0])
                    spline = interpolate.make_smoothing_spline(lane[0], lane[2], lam=10)
                    znew = spline(lane[0])
                    img_inter = add_lane_to_2d_canvas(
                        img_inter, lane[0], ynew, znew, color
                    )

                cv2.imwrite(save_path + ".png", img)
                cv2.imwrite(save_path + "_interpol.png", img_inter)


def save_bev_images(canvas, save_path, classif, c=None):
    """Saves 2 bev images with separation of classes and without"""
    seg_img = colors[canvas]
    cv2.imwrite(save_path + "_bev.png", seg_img)

    if classif:
        c = c.squeeze()[1:, :, :].argmax(0) + 1
        classes = np.zeros((c.shape[0], c.shape[1], 3), dtype=np.uint8)

        for id in np.unique(canvas)[1:]:
            color = [0, 0, 0]
            color[np.argmax(np.bincount(c[canvas == id])) - 1] = 255
            classes[canvas == id] = color

        cv2.imwrite(save_path + "_class_bev.png", classes)


def prepare_frame_gt(gt):
    """Create a list of (filtered) ground-truth lines for one frame

    Arguments:
    ----------
        lines, ??? - post-treated output of the model
        gt, Dict[str: List[Dict[str: ...]]] - dictionary of ground truth data for this frame
            'geometries' is the only important gt key for the purpose of theis function
        save_path, str - path for saving the prediction results and ground truth

    """
    frame_lanes_gt = []
    if gt is not None and len(gt.keys()) > 0:
        res_lanes = {}
        for lane_idx in range(len(gt["geometries"])):
            visibilities = gt["geometries"][lane_idx]["visibilities"]
            if not any(visibilities):
                continue
            type_id = int(gt["geometries"][lane_idx]["type_id"])
            if type_id in [0, 1, 66, 70, 997, 998]:  # ego traj
                continue
            index = int(gt["geometries"][lane_idx]["index"])

            coords = np.array(gt["geometries"][lane_idx]["body_coordinates"])
            res_points = np.array([-1 * coords[:, 1], coords[:, 0], coords[:, 2]])
            res_points = res_points[
                :,
                np.logical_or(
                    np.array(visibilities + [False], dtype=bool),
                    np.array([False] + visibilities, dtype=bool),
                ),
            ]

            res_lanes[index] = res_points

        for index, lane in res_lanes.items():
            frame_lanes_gt.append(lane.T.tolist())

    return frame_lanes_gt


def evaluate_dataset(res_save_path):
    res_list = sorted(glob(os.path.join(res_save_path, "*.json")))
    lane_eval = LaneEval()
    for item in tqdm(res_list):
        try:
            with open(item, "r") as f:
                res = json.load(f)
            logger.debug(item)

            lane_eval.bench_all(res[0], res[1])
        #  lane_eval.show()
        except Exception as e:
            logger.error(f"Raised error: {e}")
    results = lane_eval.show()
    # shutil.rmtree(path=res_save_path)
    # os.makedirs(res_save_path)
    with open(os.path.join(res_save_path, "results.txt"), "w") as f:
        f.write(model_path + "\n")
        json.dump(results, f)


if __name__ == "__main__":
    # def val():
    model = configs.model()
    model = load_model(model, model_path)
    classif = configs.classif
    logger.info(f"Model path: {model_path}")
    model.cuda()
    model.eval()
    val_dataset = configs.val_dataset()

    logger.info(f"Validation dataset length: {len(val_dataset)}")

    ############ uncomment to write train and test seq ids to txt files  ############
    # train_seqs = []
    # for id in val_dataset.train_ids:
    #     train_seqs.append(os.path.basename(val_dataset.seq_list[id]) + "\n")
    # with open(
    #     "/data/gvincent/pseudo-gt-labelling/bev_lane_det/loader/bev_road/train_seqs.txt",
    #     "w",
    # ) as f:
    #     f.writelines(train_seqs)
    # test_seqs = []
    # for id in val_dataset.test_ids:
    #     test_seqs.append(os.path.basename(val_dataset.seq_list[id]) + "\n")
    # with open(
    #     "/data/gvincent/pseudo-gt-labelling/bev_lane_det/loader/bev_road/test_seqs.txt",
    #     "w",
    # ) as f:
    #     f.writelines(test_seqs)

    ############ uncomment to subsample test data (A=1st frame, B=all **25.jpg frames) ############

    # A)
    import pandas as pd

    val_dataset.gt_seq_frameids = (
        pd.DataFrame(val_dataset.gt_seq_frameids)
        .groupby("seq_id")
        .min()
        .reset_index()
        .to_dict("records")
    )
    # or B)
    # df = pd.DataFrame(val_dataset.gt_seq_frameids)
    # val_dataset.gt_seq_frameids = (
    #     df[df["frame_number"].astype(int) % 25 == 0]
    #     .sort_values(["seq_id", "frame_number"])
    #     .to_dict("records")
    # )

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=False
    )

    """ Make temporary storage files according to time """
    time1 = int(time.time())
    # time1 = 1691412232

    if save_viz:
        viz_save_path = os.path.join(tmp_save_path, str(time1) + "_viz")
        os.makedirs(viz_save_path, exist_ok=True)
    if evaluate:
        res_save_path = os.path.join(tmp_save_path, str(time1) + "_res")
        os.makedirs(res_save_path, exist_ok=True)
    """ get model result and save"""
    for i, item in enumerate(tqdm(val_loader)):
        (
            image,
            bev_gt_segment,
            bev_gt_instance,
            bev_gt_offset,
            bev_gt_z,
            bev_gt_category,
            image_gt_segment,
            image_gt_instance,
            img_name,
            cam_extrinsics,
            cam_intrinsic,
        ) = item
        # We load the image and gt for the visualization / evaluation only
        # It's different from image tensor, which is it the virtual camera frame
        image_for_viz = []
        gt = []
        for j in range(image.shape[0]):
            # Shuffling is not allowed!
            image_id = i * BATCH_SIZE + j
            image_frame, gt_frame, _ = val_loader.dataset.load_and_undistort_if_needed(
                image_id
            )
            image_for_viz.append(image_frame)
            gt.append(gt_frame)

        treat_one_image(
            model,
            image,
            img_name,
            cam_extrinsics,
            cam_intrinsic,
            image_for_viz,
            gt,
            classif,
            viz_save_path,
            res_save_path,
        )

    if evaluate:
        evaluate_dataset(res_save_path)
