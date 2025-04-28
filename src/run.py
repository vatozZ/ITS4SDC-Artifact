import argparse
import yaml
import os
from RoadCharacteristics import ExtractRoadCharacteristics
from DatasetPreprocessing import CombineFiles
from ITS4SDC_Network import Network

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main(trained_model_file=None, test_file=None, config_path='configs/config.yaml'):

    config = load_config(config_path)

    data_dir = os.path.join(project_root, config.get('data_dir', 'data/'))

    if trained_model_file is None and test_file is not None:
        trained_model_file = get_trained_model_file(data_dir=data_dir)

    combined_dataset_filename = str(config.get('combined_dataset_filename', 'dataset_combined.json'))

    interpolated_road_points_size = config.get('interpolated_road_points_size', 197)

    combined_dataset_filename = CombineFiles(data_dir, combined_dataset_filename, interpolated_road_points_size, config)  # combine the files in the dataset within a single JSON file.

    print("Dataset is combined.")

    road_characteristics = ExtractRoadCharacteristics(feature='angles-lengths', combined_dataset_filename=combined_dataset_filename).get_road_characteristics()  # extract the road characteristics that will be used for training.

    print("The road characteristic is extracted.")

    network = Network(road_characteristics=road_characteristics, config=config, trained_model_file=trained_model_file, test_file=test_file)

    network.model_pipeline()


def get_trained_model_file(data_dir):

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.keras'):
                return os.path.join(root, file)
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False,
                        default='configs/config.yaml',
                        help="Path to experiment config file")

    parser.add_argument('--test_file', type=str, required=False, help='Test on the trained data.', default=None)

    parser.add_argument('--trained_model_file', type=str, required=False, help='Trained model.', default=None)

    args = parser.parse_args()

    trained_model_file = args.trained_model_file
    test_file = args.test_file

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config = load_config(args.config)

    main(trained_model_file=trained_model_file, test_file=test_file, config_path=args.config)
