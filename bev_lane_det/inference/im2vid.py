import glob

import cv2
from tqdm import tqdm

# import numpy as np

frameSize = (3840, 1920)

out = cv2.VideoWriter(
    "/data/gvincent/pseudo-gt-labelling/bev_lane_det/tmp_stellantis_results/infer/1696515791/output_video.avi",
    cv2.VideoWriter_fourcc(*"DIVX"),
    30,
    frameSize,
)

for filename in tqdm(
    sorted(
        glob.glob(
            "/data/gvincent/pseudo-gt-labelling/bev_lane_det/tmp_stellantis_results/infer/1696515791/viz/*interpol.png"
        )
    )
):
    img = cv2.imread(filename, -1)
    out.write(img)

out.release()
