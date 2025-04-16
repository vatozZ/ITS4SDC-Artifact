import argparse
import yaml
import os

from absl.testing.parameterized import NoTestsError

from train import train_and_validate, full_train
from evaluate import test_on_trained_data
from RoadCharacteristics import ExtractRoadCharacteristics
from DatasetPreprocessing import CombineFiles

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def main(trained_model=None, test_file=None):

    """
    Train Test
    _______________________________________________________
    None   None     TRAIN ONLY                  TRAIN
    None    True    TRAIN and TEST              TRAIN-TEST
    True    None    ----                        ----
    True    True    NO TRAIN;  ONLY TEST        TEST

    :param trained_model:
    :param test_file:
    :return:
    """

    if trained_model is None and test_file is not None:
        trained_model = get_trained_model()

    data_dir = os.path.join(project_root, config.get('data_dir', 'data/'))

    training_mode = config.get('training_mode', 'crossfold')

    combined_dataset_filename = str(config.get('combined_dataset_filename', 'dataset_combined.json'))

    interpolated_road_points_size = config.get('interpolated_road_points_size', 197)

    k_fold = config.get('k_fold', 10)  # default 10-fold cross-validation


    CombineFiles(data_dir, combined_dataset_filename,
                 interpolated_road_points_size)  # combine the files in the dataset within a single JSON file.

    combined_dataset_filename = os.path.join(data_dir, combined_dataset_filename)

    road_characteristics = ExtractRoadCharacteristics(feature='angles-lengths',
                                                      combined_dataset_filename=combined_dataset_filename).get_road_characteristics()  # extract the road characteristics that will be used for training.
    print("Training mode:", training_mode)

    # Train = None ; Test = None => TRAIN ONLY
    if test_file is None and trained_model is None:

        if training_mode == 'crossvalidate':
            _ = train_and_validate(road_characteristics=road_characteristics, k_fold=k_fold, config=config)

        elif training_mode == 'full':
            _ = full_train(road_characteristics=road_characteristics, config=config)

        else:
            raise ValueError('Undeclared training mode.')

    # Train = available and Test = Available => DO NOT TRAIN, ONLY TEST
    if test_file is not None and trained_model is not None:
        test_on_trained_data(trained_model=trained_model, test_file=test_file, data_dir=data_dir)

    # Train = None ; Test = Available: TRAIN and TEST
    if test_file is not None and trained_model == None:

        if training_mode == 'crossvalidate':
            trained_model = train_and_validate(road_characteristics=road_characteristics, k_fold=k_fold, config=config)

        elif training_mode == 'full':
            trained_model = full_train(road_characteristics=road_characteristics, config=config)

        else:
            raise ValueError('Undeclared training mode.')

        test_on_trained_data(trained_model=trained_model, test_file=test_file, data_dir=data_dir)

    if trained_model is not None and test_file is None:
        raise ValueError("The model has already trained.")

def get_trained_model():

    data_dir = 'data/'

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.h5'):
                return os.path.join(root, file)
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False,
                        default='experiments/configs/config.yaml',
                        help="Path to experiment config file")

    parser.add_argument('--test_file', type=str, required=False, help='Test on the trained data.')

    parser.add_argument('--trained_model', type=str, required=False, help='Trained model.', default=None)

    args = parser.parse_args()

    trained_model = args.trained_model
    test_file = args.test_file

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config = load_config(args.config)

    main(trained_model=trained_model, test_file=test_file)