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

    CombineFiles(data_dir, combined_dataset_filename, interpolated_road_points_size)  # combine the files in the dataset within a single JSON file.

    combined_dataset_filename = os.path.join(data_dir, combined_dataset_filename)

    road_characteristics = ExtractRoadCharacteristics(feature='angles-lengths', combined_dataset_filename=combined_dataset_filename).get_road_characteristics()  # extract the road characteristics that will be used for training.

    network = Network(road_characteristics=road_characteristics, config=config, trained_model_file=trained_model_file, test_file=test_file)

    network.model_pipeline()


    """

    # Train = None ; Test = None => TRAIN ONLY
    if test_file is None and trained_model_file is None:
        if training_mode == 'crossvalidate':
            _ = train_and_validate(road_characteristics=road_characteristics, k_fold=k_fold, config=config)

        elif training_mode == 'full':
            _ = full_train(road_characteristics=road_characteristics, config=config)

        else:
            raise ValueError('Undeclared training mode.')

    # Train = available and Test = Available => DO NOT TRAIN, ONLY TEST
    if test_file is not None and trained_model_file is not None:
        test_on_trained_data(trained_model_file=trained_model_file, test_file=test_file, data_dir=data_dir)

    # Train = None ; Test = Available: TRAIN and TEST
    if test_file is not None and trained_model_file == None:
        if training_mode == 'crossvalidate':
            trained_model_file = train_and_validate(road_characteristics=road_characteristics, k_fold=k_fold, config=config)

        elif training_mode == 'full':
            trained_model_file = full_train(road_characteristics=road_characteristics, config=config)
        else:
            raise ValueError('Undeclared training mode.')

        test_on_trained_data(trained_model_file=trained_model_file, test_file=test_file, data_dir=data_dir)

    if trained_model_file is not None and test_file is None:
        raise ValueError("The model has already trained.")
    
    """

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

    parser.add_argument('--test_file', type=str, required=False, help='Test on the trained data.')

    parser.add_argument('--trained_model_file', type=str, required=False, help='Trained model.', default=None)

    args = parser.parse_args()

    trained_model_file = args.trained_model_file
    test_file = args.test_file

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config = load_config(args.config)

    main(trained_model_file=trained_model_file, test_file=test_file, config_path=args.config)
