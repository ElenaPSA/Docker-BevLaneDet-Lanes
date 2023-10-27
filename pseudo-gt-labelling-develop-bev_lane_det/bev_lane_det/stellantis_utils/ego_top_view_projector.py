import copy
import glob
import json
import logging
import math
import os

import cv2
import numpy as np
from scipy.spatial import distance
from scipy.special import expit
from tqdm import tqdm

from bev_lane_det.models.util.cluster import (
    embedding_post_centers,
    temporal_embedding_post2,
)
from bev_lane_det.models.util.post_process import bev_instance2points_with_offset_z_bis
from bev_lane_det.stellantis_utils.utils import (
    extract_can_data,
    loadImageList,
    loadOdomList,
)
from bev_lane_det.utilities.config_util import load_config_module
from bev_lane_det.utilities.coord_util import ego2image_orig

LOG = logging.getLogger(__name__)

np.random.seed(42)
color_list = (np.random.rand(200, 3) * 255).astype(np.uint8)


DEFAULT_CONFIG = {
    "seqFormat": "aimotive",
    "seqDir": "",
    "rawLaneDir": "",
    "laneSrc": "bev_lane_det",
    "startFrame": 0,
    "stopFrame": 1000,
    "step": 1,
    "config_file": "stellantis_dataset_config.py",
    "bev_width": 250,
    "bev_height": 250,
    "ego_origin_z": 0.5,
    "historic_pool": 50,
    "max_cum_heading_deg": 90,  # stop accumlation if the ego cumulated heading angle exceeds this value
    "confThresh": 0.2,
    "confThresh_emb": 0.5,
    "post_emb_margin": 6.0,
    "chain_emb_margin": 7.0,
    "post_min_cluster_size": 15,
    "accum_strategy": "mean",
    "saveDir": "",
    "saveImgToBinary": False,
    "saveImgToPng": False,
}


class TopViewProjector:
    """This class projects 3D lanes to top view (in ego frame)
    with temporal fusion, using CAN data. Non causal computation
    is possible (it uses the future observations as well as the past).
    It is agnostic of the sequence source provided that data is
    formatted correctly.
    """

    def __init__(self, config, use_past_obs=True, use_future_obs=True):
        """
        config: a dict with the following fields:
            seqDir: the path to the folder containing the raw sequence (image1.mkv, CanEGO.raw)
            startFrame: the frame id to start from (default: 0)
            stopFrame: the frame id to end at (default: 100)
            step: the frame step (>=1) (default: 1)
            topViewWidth_px: the width of the top view in pixels (default: 400)
            topViewRes_m: the resoultion of the top view, in m/px (default: 0.1)
            ego_origin_z: the altitude of the ego frame, relative to ground (default: 0.5)
            historic_pool: the maximal nb of timestamps to use in the past and in the future (if non-causal)
        """
        self.use_past_obs = use_past_obs
        self.use_future_obs = use_future_obs
        self.config_data = DEFAULT_CONFIG
        for key, val in config.items():
            self.config_data[key] = val

        self.bev_config = load_config_module(config["config_file"])

        # Sanity checks
        if self.config_data["saveImgToBinary"] or self.config_data["saveImgToPng"]:
            assert (
                len(self.config_data["saveDir"]) > 0
            ), "You must specify a valid output dir for saving results."
            if not os.path.isdir(self.config_data["saveDir"]):
                os.makedirs(self.config_data["saveDir"])

        # Load navigation data
        assert self.config_data["seqFormat"] in [
            "aimotive",
            "jlab",
        ], "Unknown data format: {}".format(self.config_data["seqFormat"])
        if self.config_data["seqFormat"] == "jlab":
            self.read_data_jlab(config["seqDir"])
        elif self.config_data["seqFormat"] == "aimotive":
            self.read_data_aimotive(self.config_data["seqDir"])
        else:
            logging.error(
                "Unknown data format: {}".format(self.config_data["seqFormat"])
            )
        # Build top view mesh
        self.build_top_view_mesh()
        # Caching
        self.cached_scores = (
            {}
        )  # dictionnary with key = timestamp and value = 3d lanes coords
        self.cached_raw = (
            {}
        )  # dictionnary with key = timestamp and value = 3d lanes coords
        self.embedding_data = (
            {}
        )  # dictionnary with key = timestamp and value = embedding centers (from bevlanedet) + coordinates of points

        segFile_idx = list(range(len(self.rgbTimestamp[1])))
        # segFile_idx = [
        #     id
        #     for id in range(
        #         self.config_data["stopFrame"] - self.config_data["startFrame"] + 1
        #     )
        # ]
        self.lane_data_dict, self.cam_calib_dict = self.get_lane_raw_from_index(
            segFile_idx, self.config_data["rawLaneDir"]
        )
        self.raw_lane_embedding_process(segFile_idx)
        (
            self.chained_centers,
            self.chained_centers_cov,
            self.chained_pts_list,
        ) = self.compute_chained_embeddings(segFile_idx)
        return

    def build_top_view_mesh(self):
        """
        Build a mesh grid for the final BEV (ued for accumulation)
        As the final BEV must be larger than the bev provided by the lane model,
        this function also compute the row/col indices for inserting the lane model BEV into the final BEV
        """
        self.top_xmin = -self.config_data["bev_height_m"] / 2
        self.top_ymin = -self.config_data["bev_width_m"] / 2
        self.top_xmax = self.config_data["bev_height_m"] / 2
        self.top_ymax = self.config_data["bev_width_m"] / 2
        res = self.bev_config.meter_per_pixel
        width_px = self.config_data["bev_width_m"] / res
        height_px = self.config_data["bev_height_m"] / res
        self.width_px = width_px
        self.height_px = height_px
        grid_x = np.linspace(
            self.top_xmin,
            self.top_xmax,
            int((self.top_xmax - self.top_xmin) / res) + 1,
            endpoint=True,
            dtype=np.float32,
        )
        grid_y = np.linspace(
            self.top_ymin,
            self.top_ymax,
            int((self.top_ymax - self.top_ymin) / res) + 1,
            endpoint=True,
            dtype=np.float32,
        )
        self.mesh_x, self.mesh_y = np.meshgrid(grid_x, grid_y)
        self.mesh_x_center = self.mesh_x[:-1, :-1] + res / 2
        self.mesh_y_center = self.mesh_y[:-1, :-1] + res / 2

        bev_grid_y = np.linspace(
            self.bev_config.x_range[0],
            self.bev_config.x_range[1],
            int((self.bev_config.x_range[1] - self.bev_config.x_range[0]) / res) + 1,
            endpoint=True,
            dtype=np.float32,
        )
        bev_grid_x = np.linspace(
            self.bev_config.y_range[0],
            self.bev_config.y_range[1],
            int((self.bev_config.y_range[1] - self.bev_config.y_range[0]) / res) + 1,
            endpoint=True,
            dtype=np.float32,
        )
        bev_mesh_x, bev_mesh_y = np.meshgrid(
            bev_grid_x, bev_grid_y
        )  # x: left ; y: forward
        bev_mesh_x_center = bev_mesh_x[:-1, :-1] + res / 2
        bev_mesh_y_center = bev_mesh_y[:-1, :-1] + res / 2
        bev_mesh_y_center = bev_mesh_y_center[::-1, :]

        self.r_start = self.mesh_x_center.shape[0] // 2 - int(bev_grid_y[-1] / res)
        self.r_end = self.r_start + bev_mesh_y_center.shape[0]
        self.c_start = int((bev_grid_x[0] - self.mesh_x[0, 0]) / res)
        self.c_end = self.c_start + bev_mesh_x_center.shape[1]

        self.top_view_center = (int(width_px // 2), int(height_px // 2))
        self.top_view_res = res

        return

    def extract_neighbour_timestamps(self, idx, usePast=True, useFuture=True):
        """
        Given a current img idx, this function fetch the corresponding timestamp as well as all
        past and future timestamps involved in BEV time accumulation.
        The ego position at current timestamp is (0,0) in the BEV. A past or future timestamp is
        kept if the corresponding ego position falls in the BEV
        (approximated by a distance threshold from the BEV center)
        Also, the cumulated ego heading angle is computed (from present to past, and from present to future)
        Timestamp fetching is stopped as soon as the cumulated heading angle od 180° is reached.
        This avoid keeping too many timestamp in roundabouts or similar situations.

        :param usePast whether to keep past timestamps or not
        :type bool
        :param useFuture whether to keep future timestamps or not
        :type bool
        """
        # As a coarse rule, use vehicle position to compute the past and future
        # distance from the current position
        if self.config_data["seqFormat"] == "jlab":
            motion_transform = self.compute_motion_transform_jlab
        elif self.config_data["seqFormat"] == "aimotive":
            motion_transform = self.compute_motion_transform_aimotive

        min_idx = self.config_data["startFrame"]
        max_idx = self.config_data["stopFrame"]
        topViewWidth_m = self.config_data["bev_width_m"]
        max_cum_heading = self.config_data["max_cum_heading_deg"] * np.pi / 180.0
        dist_thr = topViewWidth_m * np.sqrt(2) / 2

        # Extract start index
        max_iter = 1500
        start_idx = idx
        if usePast:
            dist = 0.0
            iter = 0
            rot_z = [0.0]
            while (
                (start_idx > 0)
                & (dist < dist_thr)
                & (iter < max_iter)
                & (np.abs(rot_z[0]) < max_cum_heading)
            ):
                start_idx -= 1
                iter += 1
                rot_z, trans_ = motion_transform(idx, [start_idx])
                dist = np.sqrt(trans_[0][0] ** 2 + trans_[0][1] ** 2)

        # Extract stop index
        stop_idx = idx
        if useFuture:
            dist = 0.0
            iter = 0
            rot_z = [0.0]
            while (
                (stop_idx < len(self.rgbTimestamp[0]) - 1)
                & (dist < dist_thr)
                & (iter < max_iter)
                & (np.abs(rot_z[0]) < max_cum_heading)
            ):
                stop_idx += 1
                iter += 1

                rot_z, trans_ = motion_transform(idx, [stop_idx])
                dist = np.sqrt(trans_[0][0] ** 2 + trans_[0][1] ** 2)

        # Keep max_size timestamps max
        max_size = self.config_data["historic_pool"]
        segFileIdx = [
            idx for idx in range(start_idx, stop_idx + 1) if (idx + min_idx) <= max_idx
        ]
        keep_idx = np.linspace(0, stop_idx - start_idx, max_size, dtype=np.int64)
        keep_idx = np.unique(keep_idx).tolist()
        segFileIdx = [segFileIdx[idx] for idx in keep_idx]

        return segFileIdx

    def compute_motion_transform_jlab(self, refIdx, segFileIdx):
        """
        Compute the relative rotations and translations of the ego between the timestamp
        of the refIdx and timestamps of the segFileIdx indices.
        This function is used only with jlab sequence (where ego motion is incremental)

        :param refIdx indice of the current timestamp
        :type int
        :param segFileIdx indices of the target timestamps
        :type list of int
        """
        rot_list = []
        trans_list = []
        for ii in segFileIdx:
            # Retrieve the IMU speed for timestamp interval
            ts_ind0 = np.minimum(refIdx, ii)
            ts_ind1 = np.maximum(refIdx, ii)
            start_ts = self.rgbTimestamp[0][ts_ind0]
            stop_ts = self.rgbTimestamp[0][ts_ind1]
            speed_ts, speed_vals = self.extract_data_from_timestamps(
                self.speedTimestamp[0], self.speedTimestamp[1], start_ts, stop_ts
            )

            # Compute motion
            dx = 0.0
            dy = 0.0
            dtheta = 0.0
            if len(speed_ts) > 1:
                # Resample angularVel_vals according to speed_vals
                angularVel_vals = np.interp(
                    speed_ts, self.anglarRateTimestamp[0], self.anglarRateTimestamp[1]
                )
                if refIdx > ii:  # past observations: reverse time and data
                    speed_ts = speed_ts[::-1]
                    speed_vals = speed_vals[::-1]
                    angularVel_vals = angularVel_vals[::-1]
                dx, dy, dtheta = self.compute_kinematics2(
                    speed_ts, angularVel_vals, speed_vals, 0.0
                )

            trans_list.append([dx, dy])
            rot_list.append(dtheta)
        return rot_list, trans_list

    def compute_motion_transform_aimotive(self, refIdx, segFileIdx):
        """
        Compute the relative rotations and translations of the ego between the timestamp
        of the refIdx and timestamps of the segFileIdx indices.
        This function is used only with aimotive sequence (where ego motion is absolute)

        :param refIdx indice of the current timestamp
        :type int
        :param segFileIdx indices of the target timestamps
        :type list of int
        """
        rot_list = []
        trans_list = []

        RT_start_inv = np.eye(4)

        for ii in segFileIdx:
            ts_ind0 = refIdx
            ts_ind1 = ii
            if refIdx == ii:
                trans_list.append([0.0, 0.0])
                rot_list.append(0.0)
                continue

            RT_start = np.array(self.rgbTimestamp[2][ts_ind0])
            RT_stop = np.array(self.rgbTimestamp[2][ts_ind1])

            R_start_inv = RT_start[:3, :3].T
            T_start_inv = -R_start_inv @ RT_start[:3, -1]
            RT_start_inv[:3, :3] = R_start_inv
            RT_start_inv[:3, -1] = T_start_inv

            RT_stop_relative = RT_start_inv @ RT_stop
            trans_list.append([RT_stop_relative[0, -1], RT_stop_relative[1, -1]])
            rot_list.append(np.arctan2(RT_stop_relative[1, 0], RT_stop_relative[0, 0]))

        return rot_list, trans_list

    def read_data_jlab(self, dataPath):
        """
        Read CAN data files for jlab sequences.
        The following files should be present in dataPath:
        - 'speed.txt'
        - 'angularRate.txt'
        - 'Image1.index'
        :param dataPath path to the directory containing CAN files
        :type str
        """
        speed_path = os.path.join(dataPath, "speed.txt")
        angularRate_path = os.path.join(dataPath, "angularRate.txt")
        images_list_path = os.path.join(dataPath, "Image1.index")

        need_extraction = not (
            os.path.isfile(speed_path) and os.path.isfile(angularRate_path)
        )
        if need_extraction:
            can_file = angularRate_path = os.path.join(dataPath, "CanEGO.raw")
            LOG.info(
                "IMU speed and angular rate data files not found. They will be generated in {}.".format(
                    can_file
                )
            )
            extract_can_data(can_file)

        img, ts = loadImageList(images_list_path)
        # Extract the indice of the 1st image
        basename = img[0].split(".")[0]
        img_id_start = int(basename.split("_")[1])
        self.max_iter = len(ts)
        self.rgbTimestamp = {
            0: np.array(
                ts[self.config_data["startFrame"] : self.config_data["stopFrame"] + 1]
            ).astype(np.double),
            1: [
                os.path.join(dataPath, "Image1", elt)
                for elt in img[
                    self.config_data["startFrame"]
                    - img_id_start : self.config_data["stopFrame"]
                    - img_id_start
                    + 1
                ]
            ],
        }

        ts, val = loadOdomList(speed_path)
        self.speedTimestamp = {0: np.array(ts).astype(np.double), 1: np.array(val)}
        ts, val = loadOdomList(angularRate_path)
        self.anglarRateTimestamp = {0: np.array(ts).astype(np.double), 1: np.array(val)}

    def read_data_aimotive(self, seq_path):
        """
        Read CAN data files for aimotive sequences.
        The following files should be present in seq_path:
        - '/sensor/gnssins/egomotion2.json/egomotion2.json'
        - '/sensor/camera_undist/F_MIDRANGECAM_C/*.jpg'
        - '/dynamic/box/3d_body/*.json'
        :param seq_path path to the root directory
        :type str
        """
        ego_file = seq_path + "/sensor/gnssins/egomotion2.json/egomotion2.json"
        img_dir = seq_path + "/sensor/camera_undist/F_MIDRANGECAM_C"
        box_dir = seq_path + "/dynamic/box/3d_body/"
        box_list = sorted(glob.glob(box_dir + "*.json"))
        # box_ids_str=[elt.split('_')[-1] for elt in box_list]
        # box_ids= [int(elt.split('.')[0]) for elt in box_ids_str]

        img_list = glob.glob(img_dir + "/*.jpg")
        img_list = sorted(
            list(
                filter(
                    lambda x: box_dir + "frame_" + x.split("_")[-1][:-4] + ".json"
                    in box_list,
                    img_list,
                )
            )
        )
        img_ids_str = [elt.split("_")[-1] for elt in img_list]
        img_ids = [int(elt.split(".")[0]) for elt in img_ids_str]

        assert os.path.isfile(ego_file), "Cannot find {}".format(ego_file)

        if (img_ids[0] > self.config_data["startFrame"]) or (
            self.config_data["startFrame"] < 0
        ):
            self.config_data["startFrame"] = img_ids[0]
        else:
            img_list = [
                elt
                for elt, id in zip(img_list, img_ids)
                if id >= self.config_data["startFrame"]
            ]
            img_ids = [elt for elt in img_ids if elt >= self.config_data["startFrame"]]

        if (img_ids[-1] < self.config_data["stopFrame"]) or (
            self.config_data["stopFrame"] < 0
        ):
            self.config_data["stopFrame"] = img_ids[-1]
        else:
            img_list = [
                elt
                for elt, id in zip(img_list, img_ids)
                if id <= self.config_data["stopFrame"]
            ]
            img_ids = [elt for elt in img_ids if elt <= self.config_data["stopFrame"]]

        with open(ego_file) as f:
            ego_data = json.load(f)
        tsbox = []
        for boxfile in box_list:
            with open(boxfile) as f:
                box_data = json.load(f)
                ts = int(box_data["Timestamp"])
                tsbox.append(ts)

        RTData = []
        time_gnss = []
        for k, val in ego_data.items():
            time_host = int(val["time_host"] * 1e9)
            time_gnss.append(time_host)
            RTData.append(val)
        time_gnss = np.asarray(time_gnss)
        RTnew = []
        for t in tsbox:
            idx = np.argmin(np.abs(time_gnss - t))
            RTnew.append(RTData[idx]["RT_ECEF_body"])

        RT = RTnew
        ts = tsbox
        self.rgbTimestamp = {0: np.array(ts), 1: img_list, 2: RT}
        self.max_iter = len(ts)

    def get_lane_raw_from_index(self, fileIdx, lane_dir):
        """
        Load a list of 3d lane files + camera calibrations from a list of fileIdx indexes
        The .npy files must be generated in advance. For the bev_lane_det model,
        the generation is performed by running the inference module
        The output timestamp list contains the actually loaded scores
        :param fileIdx list of lane files indexes
        :type list of int
        :param lane_dir path to the root directory containing the npy files
        :type str
        """
        lane_data_dict = {}
        cam_calib_dict = {}
        ts_to_load = []

        if self.config_data["seqFormat"] == "jlab":
            lane_fmt = "image_{0:05d}.np.npy"
        else:
            lane_fmt = "_{0:07d}_raw_pred.npy"

        for idx in fileIdx:
            ts_to_load = np.uint64(self.rgbTimestamp[0][idx])
            if lane_data_dict.get(ts_to_load) is None:
                fileToLoad = (
                    lane_dir
                    + "/"
                    + lane_fmt.format(idx + self.config_data["startFrame"])
                )
                try:
                    data0 = np.load(fileToLoad, allow_pickle=True)
                except:
                    LOG.error("Unable to load file {}".format(fileToLoad))
                else:
                    lane_data1 = data0.item().get("pred")
                    lane_data2 = np.transpose(lane_data1[0], (1, 2, 0))
                    lane_data_dict[ts_to_load] = lane_data2
                    cam_calib_dict[ts_to_load] = {
                        "cam_intrinsics": data0.item().get("cam_intrinsics"),
                        "cam_extrinsics": data0.item().get("cam_extrinsics"),
                        "img_name": data0.item().get("img_name"),
                    }

        return lane_data_dict, cam_calib_dict

    def lane_to_top_view(self, lanes_coords, lane_cid):
        """
        Convert lane points to BEV pixel coordinates
        Returns a bgr mask with the size of the accum. BEV
        :param lanes_coords list of [list of (x,y) points, in m]
        :type list
        :param lane_cid list of lane ids, used for color selection
        :type list of int
        """
        lane_mask = np.zeros(self.mesh_x.shape, dtype=np.int32)
        for lane, cid in zip(lanes_coords, lane_cid):
            lane_xyz_m = np.array(lane)
            # convert coords to pixels
            lane_xyz_m[:, 1] -= self.top_xmin
            lane_xyz_m[:, 0] -= self.top_ymin
            lane_xy_px = lane_xyz_m[:, :2] / np.array(
                [self.top_view_res, self.top_view_res]
            )
            lane_xy_px[:, 1] = self.mesh_x.shape[0] - lane_xy_px[:, 1]
            lane_xy_px = lane_xy_px.astype(np.int64)
            lane_mask = cv2.polylines(lane_mask, [lane_xy_px], False, int(cid + 1), 1)

        return lane_mask

    def extract_data_from_timestamps(
        self, timestamps_array, values_array, start_ts, stop_ts
    ):
        """
        Loader of speed data for jlab sequences
        """
        assert start_ts <= stop_ts, "Argument start_ts must be <= stop_ts"
        if (stop_ts < timestamps_array[0]) or (start_ts > timestamps_array[-1]):
            return [], []

        timestamps = timestamps_array.reshape(-1, 1)

        idx_start = np.argmin(np.abs(timestamps - np.double(start_ts)))
        idx_stop = np.argmin(np.abs(timestamps - np.double(stop_ts)))

        ts = timestamps_array[idx_start:idx_stop].tolist() + [
            timestamps_array[idx_stop]
        ]
        vals = values_array[idx_start:idx_stop].tolist() + [values_array[idx_stop]]

        return np.array(ts), np.array(vals)

    def compute(self, iter):
        """
        Main function for accumulating lanes in BEV.
        Run the projector on image number #iter
        :param iter the image index in the timestamps list (so, starting from 0 whatever the value of image starting idx)
        :type int
        """
        # Sanity check
        assert iter >= 0, "iter argument must be a positive integer"
        assert iter < self.max_iter, "iter value ({}) exceeds max value ({})".format(
            iter, self.max_iter
        )
        if self.config_data["seqFormat"] == "jlab":
            motion_transform = self.compute_motion_transform_jlab
        elif self.config_data["seqFormat"] == "aimotive":
            motion_transform = self.compute_motion_transform_aimotive

        # According to top view width and the vehicle motion,
        # find the past and future useful idxes
        segFile_idx = self.extract_neighbour_timestamps(
            iter, usePast=self.use_past_obs, useFuture=self.use_future_obs
        )
        ts_list = [np.uint64(self.rgbTimestamp[0][idx]) for idx in segFile_idx]
        lane_data_list = [self.lane_data_dict[ts] for ts in ts_list]
        # Compute the list of transforms to apply to score_list
        rot_list, trans_list = motion_transform(iter, segFile_idx)

        # Compute accumulated mask
        top_view, line_dict = self.projectRawLanes_iterative(
            segFile_idx, lane_data_list, rot_list, trans_list
        )

        # Write results
        if self.config_data["saveImgToBinary"] or self.config_data["saveImgToPng"]:
            # Build file name
            out_file_base = self.rgbTimestamp[1][iter]
            out_file_base = os.path.basename(out_file_base)
            out_file_base = os.path.splitext(out_file_base)[0]

        # Project lines to image
        ts_now = self.rgbTimestamp[0][iter]
        img_path = self.rgbTimestamp[1][iter]
        cam_intrinsics = self.cam_calib_dict[ts_now]["cam_intrinsics"]
        cam_extrinsics = self.cam_calib_dict[ts_now]["cam_extrinsics"]
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            for lane_idx, lane in line_dict.items():
                lane = np.array(lane)
                color = tuple(color_list[lane_idx].tolist())

                uv2 = ego2image_orig(
                    np.array([lane[:, 1], -lane[:, 0], lane[:, 2]]),
                    cam_intrinsics,
                    cam_extrinsics,
                )
                img = cv2.polylines(
                    img, [uv2[0:2, :].T.numpy().astype(int)], False, color, 6
                )
            output_dir_png = os.path.join(self.config_data["saveDir"], "png")
            if not os.path.isdir(output_dir_png):
                os.makedirs(output_dir_png)
            out_file = out_file_base + "_cam.png"
            out_path = os.path.join(output_dir_png, out_file)
            cv2.imwrite(out_path, img)

        if self.config_data["saveImgToBinary"]:
            output_dir_bin = os.path.join(self.config_data["saveDir"], "binary")
            if not os.path.isdir(output_dir_bin):
                os.makedirs(output_dir_bin)
            out_file = out_file_base + ".npz"
            out_path = os.path.join(output_dir_bin, out_file)
            self.save_top_view(top_view, out_path, binary=True, bgr=False)

        if self.config_data["saveImgToPng"]:
            output_dir_png = os.path.join(self.config_data["saveDir"], "png")
            if not os.path.isdir(output_dir_png):
                os.makedirs(output_dir_png)
            out_file = out_file_base + "_bev.png"
            out_path = os.path.join(output_dir_png, out_file)
            self.save_top_view(top_view, out_path, binary=False, bgr=True)

        # Save lines to json files
        output_dir_json = os.path.join(self.config_data["saveDir"], "json")
        if not os.path.isdir(output_dir_json):
            os.makedirs(output_dir_json)
        out_file = out_file_base + ".json"
        out_path = os.path.join(output_dir_json, out_file)
        with open(out_path, "w") as f:
            json.dump(line_dict, f)

        return top_view

    def raw_lane_embedding_process(self, segFile_idx):
        """ """
        post_conf = self.config_data[
            "confThresh_emb"
        ]  # Minimum confidence on the segmentation map for clustering
        post_emb_margin = self.config_data[
            "post_emb_margin"
        ]  # embeding margin of different clusters
        post_min_cluster_size = self.config_data[
            "post_min_cluster_size"
        ]  # The minimum number of points in a cluster
        ts_list = [np.uint64(self.rgbTimestamp[0][idx]) for idx in segFile_idx]

        for ts in tqdm(ts_list):
            topview = self.lane_data_dict[ts]
            if self.embedding_data.get(ts) is None:
                # Get embeddings centers
                scores = expit(topview[:, :, 0])
                prediction = [copy.deepcopy(scores), copy.deepcopy(topview[:, :, 1:3])]
                prediction[0] = prediction[0][None, None, :, :]
                prediction[1] = np.transpose(prediction[1], (2, 0, 1))
                prediction[1] = prediction[1][None, :, :, :]
                canvas, ids, centers = embedding_post_centers(
                    prediction,
                    conf=post_conf,
                    emb_margin=post_emb_margin,
                    min_cluster_size=post_min_cluster_size,
                    canvas_color=False,
                )
                if type(centers) == list:
                    self.embedding_data[ts] = {
                        "centers": [elt[0].tolist() for elt in centers],
                        "pts": ids,
                    }
                else:
                    self.embedding_data[ts] = {
                        "centers": [centers[0].tolist()],
                        "pts": ids,
                    }

    def compute_chained_embeddings(self, segFile_idx):
        # Gather embedding centers that belong to the same lines
        emb_thr = self.config_data["chain_emb_margin"]
        ts_list = [np.uint64(self.rgbTimestamp[0][idx]) for idx in segFile_idx]
        chained_pts_list = [[] for idx in segFile_idx]

        for idx, ts in enumerate(tqdm(ts_list)):
            if idx == 0:
                chained_centers = [[elt] for elt in self.embedding_data[ts]["centers"]]
                chained_centers_cov = []
                for cent_id, cent in enumerate(self.embedding_data[ts]["centers"]):
                    pts_list = [
                        elt
                        for elt in self.embedding_data[ts]["pts"]
                        if elt[2] == cent_id
                    ]
                    pts_list_np = np.array(pts_list)
                    chained_pts_list[idx] += pts_list
                    cov_mat = (
                        np.cov(pts_list_np[:, 0], pts_list_np[:, 1]) + np.eye(2) * 1e-4
                    )  # add eps on the diag for avoiding singular matrix
                    chained_centers_cov.append([cov_mat])
            else:
                for cent_id, cent in enumerate(self.embedding_data[ts]["centers"]):
                    pts_list = [
                        elt
                        for elt in self.embedding_data[ts]["pts"]
                        if elt[2] == cent_id
                    ]
                    pts_list_np = np.array(
                        [
                            elt
                            for elt in self.embedding_data[ts]["pts"]
                            if elt[2] == cent_id
                        ]
                    )
                    cov_mat = (
                        np.cov(pts_list_np[:, 0], pts_list_np[:, 1]) + np.eye(2) * 1e-4
                    )
                    min_dist = emb_thr + 1.0
                    min_cid = -1

                    for idx_ref, cent_ref in enumerate(chained_centers):
                        try:
                            cov_mat_ref = chained_centers_cov[idx_ref][-1] + cov_mat
                            cov_mat_ref_inv = np.linalg.inv(cov_mat_ref)
                            dist = distance.mahalanobis(
                                cent, cent_ref[-1], cov_mat_ref_inv
                            )
                            # dist = distance.euclidean(cent_ref[-1], cent)
                            if len(cent_ref) > 1:
                                cov_mat_ref = chained_centers_cov[idx_ref][-2] + cov_mat
                                cov_mat_ref_inv = np.linalg.inv(cov_mat_ref)
                                dist2 = distance.mahalanobis(
                                    cent, cent_ref[-2], cov_mat_ref_inv
                                )
                                dist = np.minimum(dist, dist2)
                        except:
                            LOG.error(
                                "The inversion of lane pts cov. matrix failed at iteration {} / lane id {}".format(
                                    idx, cent_id
                                )
                            )

                        if dist < min_dist:
                            min_dist = dist
                            min_cid = idx_ref

                    if min_dist < emb_thr:
                        chained_centers[min_cid].append(cent)
                        # Update center id for points
                        pts_list_updated = [
                            (elt[0], elt[1], min_cid) for elt in pts_list
                        ]
                        chained_pts_list[idx] += pts_list_updated
                        chained_centers_cov[min_cid].append(cov_mat)
                    else:
                        min_cid = len(chained_centers)
                        chained_centers.append([cent])
                        pts_list_updated = [
                            [elt[0], elt[1], min_cid] for elt in pts_list
                        ]
                        chained_pts_list[idx] += pts_list_updated
                        chained_centers_cov.append([cov_mat])

        return chained_centers, chained_centers_cov, chained_pts_list

    def get_chained_embeddings(self, segFile_idx):
        chained_pts_list = [self.chained_pts_list[idx] for idx in segFile_idx]
        return chained_pts_list

    def projectRawLanes_iterative(
        self, segFile_idx, lane_data_list, rot_list, trans_list
    ):
        """
        This function performs the step-by-step lane accumulation of a series of timestamps in the final BEV
        :param segFile_idx list of timestamp indices involved in accumulation
        :type list of int
        :param lane_data_list list of raw lane data (provided by get_lane_raw_from_index) for each timestamp
        :type list
        :param rot_list list of ego rotation between current and each past and previous timestamps (provided by compute_motion_transform)
        :type list of float
        :param trans_list list of ego translation between current and each past and previous timestamps (provided by compute_motion_transform)
        :type list of (x,y)
        """

        post_conf = self.config_data[
            "confThresh"
        ]  # Minimum confidence on the segmentation map for clustering

        post_emb_margin = self.config_data[
            "post_emb_margin"
        ]  # embeding margin of different clusters
        post_min_cluster_size = self.config_data[
            "post_min_cluster_size"
        ]  # The minimum number of points in a cluster
        rotCenter = (self.top_view_center[1], self.top_view_center[0])

        # Apply transforms
        grid_x = np.linspace(
            0, self.width_px, int(self.width_px), endpoint=False, dtype=np.float32
        )
        grid_y = np.linspace(
            0, self.height_px, int(self.height_px), endpoint=False, dtype=np.float32
        )

        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        grid_centers_ = np.stack((mesh_x, mesh_y), axis=2)

        grid_centers = np.stack(
            (self.mesh_x_center, self.mesh_y_center[::-1, :]), axis=2
        )
        grid_offsets = np.zeros(grid_centers.shape, dtype=np.float32)
        grid_seg = np.zeros(self.mesh_x_center.shape, dtype=np.float32)

        seg_cum = np.zeros(self.mesh_x_center.shape, dtype=np.float32)
        offset_cum = np.zeros(grid_centers.shape, dtype=np.float32)
        z_pred_cum = np.zeros(self.mesh_x_center.shape, dtype=np.float32)
        weights_cum = np.zeros(self.mesh_x_center.shape, dtype=np.float32)

        emb_cum = np.zeros(grid_centers.shape, dtype=np.float32)

        topview_weights = np.ones(lane_data_list[0].shape[:-1], dtype=np.float32)
        decay_tlp = (
            np.array(
                [
                    (idx) / (lane_data_list[0].shape[0] - 1)
                    for idx in range(lane_data_list[0].shape[0])
                ]
            )
            * 2
        ) - 1  # Goes from 1 to -1

        decay = expit(decay_tlp * 6 + 1)

        topview_weights *= decay[:, None]
        grid_weights = copy.deepcopy(grid_seg)
        grid_weights[
            self.r_start : self.r_end, self.c_start : self.c_end
        ] = topview_weights

        seg = copy.deepcopy(grid_seg)
        zpred = copy.deepcopy(grid_seg)
        emb = copy.deepcopy(grid_offsets)

        embedding_data = self.get_chained_embeddings(segFile_idx)
        bev_lane_pts = []

        scale = 1.0 / self.top_view_res
        Mview = np.array(
            [
                [0, -scale, self.top_view_center[0]],
                [-scale, 0, self.top_view_center[1]],
                [0, 0, 1],
            ]
        )
        Mview_inv = np.linalg.inv(Mview)

        for idx, lane_data in enumerate(lane_data_list):
            alpha = rot_list[idx]
            # Compute transform matrix for affine warpping
            M_m = np.array(
                [
                    [math.cos(alpha), -math.sin(alpha), trans_list[idx][0]],
                    [math.sin(alpha), math.cos(alpha), trans_list[idx][1]],
                    [0, 0, 1],
                ]
            )

            M = Mview @ M_m @ Mview_inv

            grid_centers_before_warp = grid_centers_
            grid_centers_before_warp_flat = grid_centers_before_warp.reshape(1, -1, 2)
            grid_centers_before_warp_flat = grid_centers_before_warp_flat[0].T

            grid_centers_before_warp_flat = np.concatenate(
                (
                    grid_centers_before_warp_flat,
                    np.ones((1, grid_centers_before_warp_flat.shape[1])),
                ),
                axis=0,
            )
            grid_centers_after = M[0:2, :] @ grid_centers_before_warp_flat
            grid_centers_after = grid_centers_after.T
            grid_centers_after = grid_centers_after.reshape(
                grid_centers_before_warp.shape
            )
            grid_centers_after_floor = np.floor(grid_centers_after).astype(np.int64)

            grid_centers_after_floor_flat = grid_centers_after_floor.reshape(-1, 2)

            mask = (
                (grid_centers_after_floor_flat[:, 1] >= 0.0)
                * (grid_centers_after_floor_flat[:, 0] >= 0.0)
                * (grid_centers_after_floor_flat[:, 0] < grid_centers_.shape[0])
                * (grid_centers_after_floor_flat[:, 1] < grid_centers_.shape[1])
            )
            grid_centers_after_floor_flat = grid_centers_after_floor_flat[mask]

            # Apply transformation on cluster points
            pts_tmp = np.array(embedding_data[idx]).T
            pts_tmp[:2, :] = pts_tmp[:2, :][::-1]
            pts_tmp[0, :] += self.c_start
            pts_tmp[1, :] += self.r_start

            pts_tmp[-1, :] = 1

            pts_tmp2_tr = (np.floor(M @ pts_tmp)).astype(np.int64)
            pts_tr = np.array(embedding_data[idx])
            pts_tr[:, 0] = pts_tmp2_tr[1, :]
            pts_tr[:, 1] = pts_tmp2_tr[0, :]
            bev_lane_pts += pts_tr.tolist()

            # Compute weighting grid for step idx
            grid_weights_after = np.zeros_like(grid_weights).astype(np.float32)
            grid_weights_flat = grid_weights.copy().reshape(-1)
            grid_weights_flat = grid_weights_flat[mask]
            grid_weights_after[
                grid_centers_after_floor_flat[:, 1], grid_centers_after_floor_flat[:, 0]
            ] = grid_weights_flat

            # Process scores and z_pred
            seg[self.r_start : self.r_end, self.c_start : self.c_end] = lane_data[
                :, :, 0
            ]
            zpred[self.r_start : self.r_end, self.c_start : self.c_end] = lane_data[
                :, :, -1
            ]

            seg_after = np.zeros_like(seg).astype(np.float32)
            seg_flat = seg.reshape(-1)
            seg_flat = seg_flat[mask]
            seg_after[
                grid_centers_after_floor_flat[:, 1], grid_centers_after_floor_flat[:, 0]
            ] = expit(seg_flat)

            zpred_after = np.zeros_like(zpred).astype(np.float32)
            z_flat = zpred.reshape(-1)
            z_flat = z_flat[mask]

            zpred_after[
                grid_centers_after_floor_flat[:, 1], grid_centers_after_floor_flat[:, 0]
            ] = z_flat

            # Process embeddings
            emb[self.r_start : self.r_end, self.c_start : self.c_end, :] = lane_data[
                :, :, 1:3
            ]
            emb_after = np.zeros_like(emb).astype(np.float32)
            emb_flat = emb.reshape(-1, 2)
            emb_flat = emb_flat[mask]

            emb_after[
                grid_centers_after_floor_flat[:, 1], grid_centers_after_floor_flat[:, 0]
            ] = emb_flat

            # Process offsets: apply M_m to initial grid centers and M2 to initial offsets, then update
            offsetxy_ = np.zeros_like(grid_centers_before_warp).astype(np.float32)
            offset_xy = np.stack(
                (
                    lane_data[:, :, 3],
                    np.zeros(lane_data[:, :, 3].shape, dtype=np.float32),
                ),
                axis=2,
            )  # (200, 48, 2) # TODO: optimize time
            offsetxy_[self.r_start : self.r_end, self.c_start : self.c_end] = offset_xy

            offset_xy_flat = offsetxy_.reshape(1, -1, 2)

            offset_xy_after = M[0:2, 0:2] @ offset_xy_flat[0].T
            offset_xy_after = offset_xy_after.T

            grid_centers_before_warp_flat = grid_centers_before_warp.reshape(1, -1, 2)
            grid_centers_before_warp_flat = grid_centers_before_warp_flat[0].T

            grid_centers_before_warp_flat = np.concatenate(
                (
                    grid_centers_before_warp_flat,
                    np.ones((1, grid_centers_before_warp_flat.shape[1])),
                ),
                axis=0,
            )
            grid_centers_after = M[0:2, :] @ grid_centers_before_warp_flat
            grid_centers_after = grid_centers_after.T
            grid_centers_after = grid_centers_after.reshape(
                grid_centers_before_warp.shape
            )
            grid_centers_after_floor = np.floor(grid_centers_after)
            delta = grid_centers_after - grid_centers_after_floor
            grid_centers_after_floor = grid_centers_after_floor.astype(np.int64)

            grid_offset_after = np.zeros_like(grid_centers_after_floor).astype(
                np.float32
            )
            grid_centers_after_floor_flat = grid_centers_after_floor.reshape(-1, 2)
            mask = (
                (grid_centers_after_floor_flat[:, 1] >= 0.0)
                * (grid_centers_after_floor_flat[:, 0] >= 0.0)
                * (grid_centers_after_floor_flat[:, 0] < grid_offset_after.shape[0])
                * (grid_centers_after_floor_flat[:, 1] < grid_offset_after.shape[1])
            )
            grid_centers_after_floor_flat = grid_centers_after_floor_flat[mask]
            delta = delta.reshape(-1, 2)
            delta = delta[mask]
            offset_xy_after = offset_xy_after[mask]
            grid_offset_after[
                grid_centers_after_floor_flat[:, 1], grid_centers_after_floor_flat[:, 0]
            ] = (delta + offset_xy_after)

            # Cumulate
            weights_cum += grid_weights_after
            if self.config_data["accum_strategy"] == "max":
                seg_cum_mask = seg_after > seg_cum
                seg_cum[seg_cum_mask] = seg_after[seg_cum_mask]
                emb_cum[seg_cum_mask] = emb_cum[seg_cum_mask]
                offset_cum[seg_cum_mask] = grid_offset_after[seg_cum_mask]
                z_pred_cum[seg_cum_mask] = z_pred_cum[seg_cum_mask]
            elif self.config_data["accum_strategy"] == "mean":
                seg_cum += seg_after * grid_weights_after
                emb_cum += emb_after * grid_weights_after[:, :, None]
                offset_cum += grid_offset_after * grid_weights_after[:, :, None]
                z_pred_cum += zpred_after * grid_weights_after
            else:
                logging.error(
                    "Unknown accumulation strategy: {}".format(
                        self.config_data["accum_strategy"]
                    )
                )

            # ##### DEBUG #####
            # seg_cum_disp = copy.deepcopy(seg_cum)
            # weights_cum_disp = copy.deepcopy(weights_cum)
            # weights_cum_disp[weights_cum_disp<=0.] = 1.
            # seg_cum_disp /= weights_cum_disp
            # seg_cum_disp[seg_cum_disp<post_conf]=-1000
            # seg_cum_disp *= 100.
            # seg_cum_disp += 100.
            # seg_cum_disp[seg_cum_disp>255.]=255.
            # seg_cum_disp[seg_cum_disp<0.0]=0.0
            # cv2.drawMarker(seg_cum_disp, rotCenter, 255, cv2.MARKER_TRIANGLE_UP)
            # file_name = "seg_cum{0:02d}.png".format(idx)
            # cv2.imwrite(file_name, seg_cum_disp.astype(np.uint8))
            # #################

        if self.config_data["accum_strategy"] == "mean":
            weights_cum[weights_cum <= 0.0] = 1.0
            seg_cum /= weights_cum
            emb_cum /= weights_cum[:, :, None]
            offset_cum /= weights_cum[:, :, None]
            z_pred_cum /= weights_cum

        # Post-process embeddings
        seg_tmp = seg_cum[None, None, :, :]
        emb_tmp = np.transpose(emb_cum, (2, 0, 1))
        emb_tmp = emb_tmp[None, :, :, :]
        prediction = (seg_tmp, emb_tmp)

        seen = set()
        bev_lane_pts_uniq = [
            x for x in bev_lane_pts if tuple(x) not in seen and not seen.add(tuple(x))
        ]
        # remove points with out-of-bev coordinates
        max_h, max_w = self.mesh_x_center.shape
        bev_lane_pts_uniq = [
            elt
            for elt in bev_lane_pts_uniq
            if elt[0] >= 0 and elt[1] >= 0 and elt[0] < max_h and elt[1] < max_w
        ]

        canvas, ids = temporal_embedding_post2(
            prediction,
            bev_lane_pts_uniq,
            conf=post_conf,
            emb_margin=post_emb_margin,
            min_cluster_size=post_min_cluster_size,
            canvas_color=False,
        )
        lines_tmp = bev_instance2points_with_offset_z_bis(
            canvas,
            prediction,
            max_x=self.top_xmax,
            meter_per_pixal=(
                self.bev_config.meter_per_pixel,
                self.bev_config.meter_per_pixel,
            ),
            offset_y=offset_cum,
            Z=z_pred_cum,
        )

        lines = []
        for line in lines_tmp:
            res = [
                [-float(y), float(x), float(z)]
                for x, y, z in zip(line[0], line[1], line[2])
            ]
            lines.append(res)
        line_ids = np.unique(canvas[canvas > 0])

        # Draw mask from lines
        center_ids = np.unique(canvas[canvas > 0])
        cum_top_view_mask = self.lane_to_top_view(lines, center_ids)

        # Draw the ego position
        cv2.drawMarker(cum_top_view_mask, rotCenter, 99, cv2.MARKER_TRIANGLE_UP)
        line_dict = {int(key): val for key, val in zip(line_ids, lines)}
        return cum_top_view_mask, line_dict

    def save_top_view(self, topview, outpath, binary=True, bgr=False):
        """
        Save a bgr topview lane mask from a mask of indices
        """
        if bgr:
            topview_bgr = np.zeros(
                (topview.shape[0], topview.shape[1], 3), dtype=np.uint8
            )

            for label_id in np.unique(topview):
                if label_id > 0:
                    topview_bgr[topview == label_id, :] = color_list[label_id]

            topview = topview_bgr
        topview = topview.astype(np.uint8)
        if binary:
            np.savez(outpath, out=topview)
        else:
            cv2.imwrite(outpath, topview)
        return

    def compute_kinematics2(
        self, timestamps, angular_velocities, vehicle_speeds, theta
    ):
        """
        :param timestamps timestamps in us
        :type int
        :param angular_velocities list of yaw velocities, in rad/s
        :type float
        :param vehicle_speeds list of speed values, in m/s
        :type  float
        :param theta initial vehicle heading
        :type  float
        Assume the x axis is along the vehicle heading, the y axis is perpendicular, left side,
        and z is upward
        """
        assert len(timestamps) == len(
            angular_velocities
        ), "Input lists must have the same length"
        assert len(timestamps) == len(
            vehicle_speeds
        ), "Input lists must have the same length"
        assert len(timestamps) > 1, "Input lists must have at least 2 values"
        DBL_EPSILON = 1e-4

        dtheta = theta
        speed = vehicle_speeds[1:] * 1000.0 / 3600.0

        yawrate = angular_velocities[1:] * np.pi / 180.0 + 1e-5
        dt = (timestamps[1:] - timestamps[:-1]) / (1e6)

        yawrate_ok = yawrate.copy()
        yawrate_ok[speed <= DBL_EPSILON] = 0.0
        dtheta_cumsum = np.cumsum(dt * yawrate_ok) + dtheta

        dx = (1.0 / yawrate) * speed * np.sin(dt * yawrate)
        dy = (1.0 / yawrate) * speed * (1.0 - np.cos(dt * yawrate))

        dx[yawrate <= DBL_EPSILON] = (
            dt[yawrate <= DBL_EPSILON] * speed[yawrate <= DBL_EPSILON]
        )
        dy[yawrate <= DBL_EPSILON] = 0.0

        delta_x = np.sum(dx * np.cos(dtheta_cumsum) - dy * np.sin(dtheta_cumsum))
        delta_y = np.sum(dx * np.sin(dtheta_cumsum) + dy * np.cos(dtheta_cumsum))

        return delta_x, delta_y, dtheta_cumsum[-1]
