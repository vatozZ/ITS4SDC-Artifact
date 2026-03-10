import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
from RoadCharacteristics import ExtractRoadCharacteristics
from DatasetPreprocessing import CombineFiles
from ITS4SDC_Network import Network

def load_config(path):
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main(config_path, project_root):

    config = load_config(config_path)

    data_dir = os.path.join(project_root, config.get('data_dir', 'data/'))

    use_existing_model = config.get('use_existing_model', False)

    combined_dataset_filename = str(config.get('combined_dataset_filename', 'dataset_combined.json'))

    interpolated_road_points_size = config.get('interpolated_road_points_size', 197)

    onnx_model_file = config.get('onnx_model_file', None)

    combined_dataset_filename = CombineFiles(data_dir, combined_dataset_filename, interpolated_road_points_size, config)  # combine the files in the dataset within a single JSON file.

    road_characteristics = ExtractRoadCharacteristics(feature='angles-lengths', combined_dataset_filename=combined_dataset_filename).get_road_characteristics()  # extract the road characteristics that will be used for training.

    network = Network(road_characteristics=road_characteristics, config=config, project_root=project_root, use_existing_model=use_existing_model)

    if use_existing_model and os.path.exists(onnx_model_file) is not None:
        # if onnx use_existing_model flag is True, run prediction directly.
        network.run_prediction()

    else:
        # run the model pipeline
        network.model_pipeline()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False,
                        default='configs/config.yaml',
                        help="Path to experiment config file")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config = load_config(args.config)

    main(config_path=args.config, project_root=project_root)
