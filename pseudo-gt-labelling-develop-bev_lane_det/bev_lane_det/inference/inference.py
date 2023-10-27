import os
from glob import glob
from loguru import logger
from tools.val_stellantis_gt import treat_one_image, evaluate_dataset
from tqdm import tqdm
from yaml.loader import SafeLoader
from datatools.loader import Loader
from bev_lane_det.inference.plain_image_dataset import PlainImageDataset

import time
from models.util.load_model import load_model
from torch.utils.data import DataLoader
import yaml
import bev_lane_det.tools.stellantis_gt_config as train_config


MANDATORY_KEYS = {
    "model_path",
    "output_path",
    "data_folders",
}


DEFAULT_CONFIG = {
    "evaluate": True,
    "save_viz": True,
    "dataset": "aimotive",
    "use_preprocessed": False,
    "test_set_only": False,
    "default_test_ratio": 0.2,
    "batch_size": 1,
    "single_sequence": False,
    # Train configurations
    "x_range": list(train_config.x_range),
    "y_range": list(train_config.y_range),
    "meter_per_pixel": train_config.meter_per_pixel,
    "input_shape": list(train_config.input_shape),
    "output_2d_shape": list(train_config.output_2d_shape),
    "roadside": train_config.roadside,
    "motorway_only": train_config.motorway_only,
    "vc_config": train_config.vc_config,
    "classif": train_config.classif,
}


def add_default_config(config):
    """ Check that config dictionary contains mandatory keys 
    then fill in the missing keys with default values.
    Most of the default values are copied from the train config
    and should be changed only if one feels really confident.

    Arguments:
    ----------
        config: Dict[str: ...] - config dictionary as read from the input file

    Returns:
    --------
        config: Dict[str: ...] - same config dictionary, completed by default/train values
    """
    missing_keys = MANDATORY_KEYS - set(config.keys())
    if len(missing_keys) > 0:
        raise KeyError(f"The following keys are missing in the config dict: {missing_keys}")

    default_keys = set(DEFAULT_CONFIG.keys()) - set(config.keys())
    if len(default_keys) > 0:
        logger.info(f"Default values are used for the following keys: {default_keys}")
    for key in default_keys:
        config[key] = DEFAULT_CONFIG[key]

    return config


class LineInference:
    def __init__(self, config):
        self.config = add_default_config(config)
        self.model = self.load_model()
        self.root_path, self.viz_save_path, self.res_save_path = self.create_output_folders()

        self.test_ratio = 1.0  # In case of StellantisDataset, use the entire set with no splitting
        self.gt_type = None  # For AImotive data, the member will be configured later

        self.cam_calibration = None  # Can be loaded later if needed

        config_save_path = os.path.join(self.root_path, 'config.yaml')
        with open(config_save_path, 'w') as fid:
            yaml.dump(self.config, fid, default_flow_style=False)

    def load_calibration(self):
        """ Load calibration from the file. 
        Currently only AImotive calibration format is supported, others should be added.
        """
        if 'calibration_path' not in self.config or not os.path.isfile(self.config['calibration_path']):
            raise FileNotFoundError(
                f"Camera calibration file not found or not specified: {self.config['calibration_path']}"
            )
        else:
            logger.info(f"Loading camera calibration parameters for all images: {self.config['calibration_path']}")
            camera = self.read_aimotive_calibration(self.config['calibration_path'])
            # Other formats of calibration files can be added here. For example, via try-except statements
        return camera

    @staticmethod
    def read_aimotive_calibration(calibration_path):
        """ Read aimotive calibration file similarly to how it's done in the Loader class
        """
        with open(calibration_path) as fid:
            data = yaml.load(fid, Loader=SafeLoader)
        sensorconfig = list(
            filter(lambda x: x["label"] == 'F_MIDRANGECAM_C', data["sensors"])
        )[0]
        R, T = Loader.getCameraExtrinsics(sensorconfig)
        K, D, xi = Loader.getCameraIntrinsics(sensorconfig)
        camera = {'K': K, 'R': R, 'T': T, 'D': D, 'xi': xi}
        return camera
    
    def configure_aimotive_options(self):
        """ Configure usecase options for the aimotive data: 
         - whether the raw or already preprocessed data should be used 
         - whether all the folders or only the test set (20%) should be used
        """
        if self.config['use_preprocessed']:
            logger.info(
                "Preprocessed images are requested. This may result in filtering out some sequences"
            )
            self.gt_type = 'lane_postprocessed'
            if self.config['test_set_only']:
                if self.config['single_sequence']:
                    logger.error("Single sequences are not compatible with train-test split. Aborting.")
                    raise ValueError
                self.test_ratio = self.config['default_test_ratio']
                logger.info(
                    f"{self.test_ratio}-train test split will be used. Make sure you control the dataset."
                )
        else:
            logger.info("Using raw images and applying undistort on them.")
            self.gt_type = 'no_gt'
            if self.config['test_set_only']:
                logger.error("test_set option is only available for preprocessed dataset. Aborting.")
                raise ValueError

            if self.config['evaluate'] and self.gt_type == 'no_gt':
                logger.info("Evaluation will not be done as ground truth is not provided")
                self.config['evaluate'] = False

    def configure_images_options(self):
        """ Configure usecase options for the plain images in a folder data: 
        """
        if self.config['evaluate']:
            logger.info("Evaluation will not be done as ground truth is not available in the 'images' dataset")
            self.config['evaluate'] = False
        if self.config['test_set_only']:
            logger.info("Train-test split is not available for the images dataset")
        if self.config['use_preprocessed']:
            logger.info("Preprocessed images are not available for the images dataset")

    def load_data(self, data_folder):
        """ Create data generator depending on the dataset requested in the config.
        Currently only AImotive and simple-image datasets are supported.
        AImotive data format - follows the data structure provided by Stellantis.

        This function calls downstream loader functions, which must return the generator

        May return data in batches, so all elements are tensors / lists with 
        length, equal to the batch size.

        Returns:
        --------
            generator: Tuple(image, img_name, cam_extrinsics, cam_intrinsic, gt, image_for_viz)
                image, torch.Tensor - direct input to the model after all preproc
                img_name, List[str] - paths to original read images
                cam_extrinsics, List[Dict[...]] - camera parameters for every image
                    (note that images may come from different folders/sequences with 
                    different camera parameters)
                cam_extrinsics, List[Dict[...]] - ...
                image_for_viz, List[np.array] - images in the camera reference frame for saving the output 
                gt - List[Dict[...]] or List[None] - ground truth of the images if available
        """
        # First select the dataset / use case

        if self.config['dataset'] == 'aimotive':
            self.configure_aimotive_options()
            return self.aimotive_data_generator(data_folder)

        elif self.config['dataset'] == 'images':
            self.cam_calibration = self.load_calibration()
            return self.images_data_generator(data_folder)
        else:
            raise NotImplementedError(f"Loading from {self.config['dataset']} dataset is not available")

    def aimotive_data_generator(self, data_folder):
        """ Data generator for the datasets in the aimotive data format 
        The data structure follows the folder structure as given by Stellantis
        Can be used for reading both raw frames and pretreated frames.

        Returns:
        --------
            generator: Tuple(image, img_name, cam_extrinsics, cam_intrinsic, gt)
        """
        # 1. Reuse the code for the Dataset and the DataLoader 
        dataset = train_config.val_dataset(
            dataset_path=data_folder,
            gt_type=self.gt_type, 
            transformations=train_config.val_transforms,
            test_ratio=self.test_ratio,
            one_sequence=self.config['single_sequence'],
            )
        logger.info(f"Data folder: {data_folder}, dataset length: {len(dataset)}")
        # Shuffling in the DataLoader is not allowed!
        val_loader = DataLoader(
            dataset=dataset, batch_size=self.config['batch_size'], num_workers=1, shuffle=False
        )
        # 2. Main loop for the generator
        for i, frame in enumerate(tqdm(val_loader)):
            if self.gt_type == 'no_gt':
                image, img_name, cam_extrinsics, cam_intrinsic = frame
            elif self.gt_type == 'lane_postprocessed':
                image, _, _, _, _, _, _, _, img_name, cam_extrinsics, cam_intrinsic = frame
            else:
                raise NotImplementedError
            
            # Load additional data bits, that are not provided by the standard DataLoader:
            # 1) images: DataLoader returns image in the virtual camera frame, while for the
            # vizualization we need the original or rectified image. Load it.
            # 2) Ground Truth is given in a not very usable format, reload it.
            image_for_viz = []
            gt = []
            for j in range(self.config['batch_size']):
                image_id = i * self.config['batch_size'] + j
                image_frame, gt_frame, _ = val_loader.dataset.load_and_undistort_if_needed(image_id)
                image_for_viz.append(image_frame)
                gt.append(gt_frame)

            yield image, img_name, cam_extrinsics, cam_intrinsic, image_for_viz, gt

    def images_data_generator(self, data_folder):
        """ Load images for model inference for the case of the simple folder with images
        """
        image_dataset = PlainImageDataset(
            data_folder,
            self.cam_calibration,
            self.config["vc_config"],
            data_transform=train_config.val_transforms,
            return_format='full',
        )
        val_loader = DataLoader(
            dataset=image_dataset, batch_size=self.config['batch_size'], num_workers=1
        )
        for frame in val_loader:
            gt = [None for _ in range(self.config['batch_size'])]
            yield *frame, gt

    def create_output_folders(self):
        """ In the folder from the config, create the structure of the folders
        for saving the results and images
        """
        time1 = int(time.time())
        root_folder = os.path.join(self.config['output_path'], str(time1))
        viz_save_path = None
        os.makedirs(root_folder, exist_ok=True)
        if self.config['save_viz']:
            viz_save_path = os.path.join(root_folder, "viz")
            os.makedirs(viz_save_path, exist_ok=True)
        res_save_path = None
        if self.config['evaluate']:
            res_save_path = os.path.join(root_folder, "res")
            os.makedirs(res_save_path, exist_ok=True)
        return root_folder, viz_save_path, res_save_path

    def run_one_sequence(self, data_generator):
        """ get model result and save"""
        for data_batch in data_generator:
            treat_one_image(
                self.model, *data_batch,
                self.config['classif'], self.viz_save_path, self.res_save_path
            )

        if self.config['evaluate']:
            evaluate_dataset(self.res_save_path)

    def load_model(self):
        """ Load the model checkpoint. Use the train config to deduce the class of the model 
        and the path to the exact checkpoint
        """
        if not os.path.isfile(self.config['model_path']):
            raise FileNotFoundError(f"Checkpoint does not exist {self.config['model_path']}")
        model = train_config.model()
        model = load_model(model, self.config['model_path'])
        model.cuda()
        model.eval()

        logger.info(f"Loaded model: {self.config['model_path']}")
        return model

    def data_pretreatment(self):
        pass

    def post_treatment(self):
        pass
