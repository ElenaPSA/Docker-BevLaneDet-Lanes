import os
import sys
import time

sys.path.append("/data/gvincent/pseudo-gt-labelling/bev_lane_det/")
sys.path.append("/data/gvincent/pseudo-gt-labelling/")

from tqdm import tqdm
from utilities import PROJECT_DIR

from bev_lane_det.stellantis_utils.ego_top_view_projector import TopViewProjector


def main(config):
    projector = TopViewProjector(config, use_past_obs=True, use_future_obs=True)

    # projector.build_fake_3Dline_files()
    startFrame = projector.config_data["startFrame"]
    stopFrame = projector.config_data["stopFrame"]
    for iter in tqdm(range(stopFrame - startFrame + 1)):
        projector.compute(iter)


if __name__ == "__main__":
    config = {
        "seqFormat": "aimotive",  # 'aimotive' (2023 sequences) or 'jlab' (2022 or sooner sequences - NOT TESTED YET)
        # "seqDir": "/data3/VisionDatabases/PSA_2023_dataset/gt_anonymized/20230309-150222-01.02.00-01.02.15@Rikardo",
        "seqDir": "/data3/VisionDatabases/PSA_2023_dataset/gt_anonymized/20230309-150222-01.00.00-01.00.15@Rikardo",
        "rawLaneDir": "./bev_lane_det/tmp_stellantis_results/infer/1697448404/res",
        "laneSrc": "bev_lane_det",  # used for selecting the approriate lane data loader
        "startFrame": -1,  # start frame idx, or -1 to start with the smallest idx
        "stopFrame": -1,  # stop frame idx, or -1 to start with the biggest idx
        "config_file": "./bev_lane_det/tools/stellantis_gt_config.py",
        "bev_width_m": 250,  # width of the final accumulated BEV, in m (must be larger than the BEV provided by the lane model)
        "bev_height_m": 250,  # height of the final accumulated BEV, in m (must be larger than the BEV provided by the lane model)
        "ego_origin_z": 0.0,
        "historic_pool": 90,  # max number of frames used for BEV accumulation at a given timestamp
        "max_cum_heading_deg": 90,
        "confThresh": 0.0,  # threshold for sigmoid scores ; used at final stage (after accumulation)
        "confThresh_emb": 0.3,  # threshold for sigmoid scores ; used at preprocessing stage (cluster chaining)
        "chain_emb_margin": 6.0,  # margin used for temporal chaining of embedding clusters
        "post_emb_margin": 6.0,
        "post_min_cluster_size": 10,  # discard accumulated lines if they contains less points than this threshold
        "accum_strategy": "mean",
        "saveDir": "./bev_lane_det/tmp_stellantis_results/infer/1697448404/time",
        "saveImgToBinary": False,  # save BEV img of accumulated lines in binary format
        "saveImgToPng": True,  # save BEV img of accumulated lines in png format
    }

    # config = {  'seqFormat': 'jlab', # 'aimotive' (2023 sequences) or 'jlab' (2022 or sooner sequences)
    #             'seqDir': '/data2/PSA_2022/DataEcon_images/20220217_110707_Rec_JLAB09',
    #             'rawLaneDir': os.path.join(DATA_DIR, 'tmp/20220217_110707_Rec_JLAB09/1689942772_np'),
    #             'laneSrc': 'bev_lane_det',
    #             'startFrame': 4200,
    #             'stopFrame': 4500,
    #             'lane_source': 'bev_lane_det',
    #             'step': 1,
    #             'config_file': os.path.join(PROJECT_DIR, 'tools/stellantis_dataset_config.py'),
    #             'bev_width_m': 250,
    #             'bev_height_m': 250,
    #             'ego_origin_z': 0.0,
    #             'thickness':1,
    #             'minTopViewLaneHit':5,
    #             'historic_pool': 100,
    #             'max_cum_heading_deg': 90,
    #             'confThresh': 0.5,
    #             'post_emb_margin': 6.0,
    #             'post_min_cluster_size': 15,
    #             'saveDir':'/data/jdefretin/PSA_2023/pseudo-gt-labelling/bev_lane_det/data/lane_cum_top_view_output/20220217_110707_Rec_JLAB09',
    #             'saveImgToBinary':False,
    #             'saveImgToPng':True
    #         }
    main(config)
