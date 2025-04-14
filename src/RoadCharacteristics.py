import json
import numpy as np

class ExtractRoadCharacteristics:
    def __init__(self, feature, combined_dataset_filename):
        self.feature = feature
        self.test_suite = combined_dataset_filename

    def get_segment_angle_changes(self):
        pass

    def get_segment_lengths(self):
        pass

    def get_road_characteristics(self):

        with open(self.test_suite, 'r') as f:
            testSuite = json.load(f)

        segment_angles_all_list = []
        segment_lengths_all_list = []
        test_outcome_all_list = []

        for test_case in testSuite:
            roadPointsArray = np.array(test_case['road_points'])

            #1) Calculate the segment angles

            dx = roadPointsArray[1:, 0] - roadPointsArray[:-1, 0]
            dy = roadPointsArray[1:, 1] - roadPointsArray[:-1, 1]

            raw_angles = np.rad2deg(np.arctan2(dy, dx))
            segment_angles = np.zeros_like(raw_angles)
            segment_angles[1:] = np.diff(raw_angles)

            segment_angles_all_list.append(list(segment_angles))

            # 2) Calculate the segment lengths

            _diff = roadPointsArray[1:] - roadPointsArray[:-1]
            segment_lengths = list(np.linalg.norm(_diff, axis=1))

            segment_lengths_all_list.append(segment_lengths)

            # 3) Assign Test Outcomes (Labels)

            if test_case['test_outcome'] == 'FAIL':
                test_outcome_all_list.append(0)
            elif test_case['test_outcome'] == 'PASS':
                test_outcome_all_list.append(1)
            else:
                raise ValueError('Test case has not executed yet. Please execute test case to get test outcome.')

        filename = self.test_suite.split('.json')[0] + '_road_characteristics.json'

        road_characteristics = \
            {'segment_angles': segment_angles_all_list,
                'segment_lengths': segment_lengths_all_list,
                'labels': test_outcome_all_list}

        with open(filename, 'w') as f:
            json.dump(road_characteristics, f, indent=4)

        return road_characteristics