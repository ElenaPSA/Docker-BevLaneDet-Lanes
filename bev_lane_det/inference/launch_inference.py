import os
import fire
import yaml
from yaml.loader import SafeLoader
from inference import LineInference


def main(config_path):
    """Launch inference script in the lines

    Arguments:
    ----------
        config_path, str - path to .yaml file with the inference configuration
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    inference = LineInference(config)

    for data_unit in config['data_folders']:
        data_generator = inference.load_data(data_unit)
        inference.run_one_sequence(data_generator)
        inference.post_treatment()


if __name__ == '__main__':
    fire.Fire(main)
