from glob import glob
from datatools.loader import Loader
import itertools
import cv2
import numpy as np
from torch.utils.data import Dataset
from bev_lane_det.loader.bev_road.stellantis_gt_visible_data import VirtualCamera


class PlainImageDataset(Dataset):
    """ Class for loading plain distorted images for inference, located in one folder"""
    def __init__(
            self, 
            data_folder, 
            cam_calibration, 
            virtual_camera_config, 
            data_transform=None, 
            return_format=False
        ):
        self.data_folder = data_folder
        self.calibration = cam_calibration
        self.vc = VirtualCamera(virtual_camera_config, self.calibration)
        self.cam_intrinsics = np.asarray(self.calibration['K'])
        self.cam_extrinsics = self.vc.project_c2g
        self.return_format = return_format
        self.trans_image = data_transform

        self.image_files = self.list_and_filter_all_images()

    def list_and_filter_all_images(self):
        """ Look for all images in the given folder, filters and lists them for further use
        """
        # Other extensions can be added if needed
        extensions = ['.jpg', '.png']
        image_files = list(itertools.chain.from_iterable(
            [glob(f"{self.data_folder}/*{ext}") for ext in extensions]
        ))
        image_files = self.filter_image_list(image_files)

        return image_files
    
    def filter_image_list(self, image_files):
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """ To use an image in the inference we need to:
         - load it
         - undistort it
         - transform to the virtual camera frame
         - apply model-input-specific transformations

        Return an image and a undistorted image (for visualizing purposes)
        """
        img_path = self.image_files[idx]
        raw_image = cv2.imread(img_path)
        image = Loader.getUndistortedImage(raw_image, self.calibration['K'], self.calibration['D'], self.calibration['xi'])
        undist_image = image.copy()

        if self.vc.use_virtual_camera:
            image = self.vc.vitrual_camera_transform(image, self.calibration)

        # Apply transformations
        if self.trans_image is not None:
            transformed = self.trans_image(image=image)
            image = transformed["image"]

        if self.return_format == 'image_only':
            return image
        elif self.return_format == 'full':
            return image, img_path, self.cam_extrinsics, self.cam_intrinsics, undist_image
        else:
            raise NotImplementedError

