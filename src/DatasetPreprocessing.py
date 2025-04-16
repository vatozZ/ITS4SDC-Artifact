
from scipy.interpolate import interp1d
import json, os
import numpy as np
import pandas as pd
from tqdm import tqdm

def adjust_array_size(array, target_size):
    if array.shape[0] == target_size:
        return array

    elif array.shape[0] > target_size:
        indices = np.linspace(0, array.shape[0] - 1, target_size, dtype=int)
        return array[indices]


    else:
        current_indices = np.linspace(0, array.shape[0] - 1, array.shape[0])
        target_indices = np.linspace(0, array.shape[0] - 1, target_size)

        interpolator_x = interp1d(current_indices, array[:, 0], kind='linear')
        interpolator_y = interp1d(current_indices, array[:, 1], kind='linear')

        interpolated_x = interpolator_x(target_indices)
        interpolated_y = interpolator_y(target_indices)

        return np.column_stack((interpolated_x, interpolated_y))

def CombineFiles(data_dir, combined_dataset_filename, interpolated_road_points_size):

    data_list = []

    data_full_path = os.path.join(data_dir, 'executed-10000')

    if os.path.exists(os.path.join(data_dir, combined_dataset_filename)):
        return

    # iterate all files in the directory
    for file in tqdm(os.listdir(data_full_path)):
        if file.endswith('.json'):
            data_dict = {}
            # open the json file
            with open(os.path.join(data_full_path, file), 'r') as f:

                jsonfile = json.load(f)

                if 'interpolated_points' in jsonfile.keys():
                    data_dict['road_points'] = adjust_array_size(np.array(jsonfile['interpolated_points']), target_size=interpolated_road_points_size)

                elif 'interpolated_road_points' in jsonfile.keys():
                    data_dict['road_points'] = adjust_array_size(np.array(jsonfile['interpolated_road_points']), target_size=interpolated_road_points_size)

                data_dict['test_outcome'] = jsonfile['test_outcome']

            data_list.append(data_dict)

    data_frame = pd.DataFrame(data_list)

    # write out the dataframe as a JSON file.
    data_frame.to_json(os.path.join(data_dir, combined_dataset_filename), orient='records', indent=2)













