"""
This script compares the prediction speed of ONNX and KERAS models.
Runs each method 10 times and logs results to CSV with averages.
"""

import time
import tensorflow as tf
import onnxruntime
import numpy as np
import json
import csv

# provide the onnx model file
onnx_model_path = '../data/onnx_models/its4sdc.onnx'
onnxruntime_session = onnxruntime.InferenceSession(onnx_model_path)

# provide the keras model file
keras_model_path = '../data/results/result_2025_07_24_12_00/model_files/model_fold_1.keras'
keras_model = tf.keras.models.load_model(keras_model_path)


# Get the calculated road characteristic
with open('../data/dataset_combined_road_characteristics.json', 'r') as f:
    road_characteristics = json.load(f)

length_data = road_characteristics['segment_lengths']
label_data = road_characteristics['labels']
angle_data = road_characteristics['segment_angles']

features = []
for angles, lengths in zip(angle_data, length_data):
    segment_features = np.column_stack((angles, lengths))
    features.append(segment_features)

features = np.array(features).astype(np.float32)

# CSV setup
csv_filename = "../data/results/prediction_benchmark.csv"
csv_headers = ["Run", "ONNX Single", "Keras Single", "ONNX Batch", "Keras Batch"]
rows = []

onnx_single_times = []
keras_single_times = []
onnx_batch_times = []
keras_batch_times = []

for run in range(1, 11):
    # ONNX single
    t0 = time.perf_counter()
    for i in range(features.shape[0]):
        test_case = features[i].reshape(1, -1, 2)
        _ = onnxruntime_session.run(None, {onnxruntime_session.get_inputs()[0].name: test_case})
    t1 = time.perf_counter()
    onnx_single_time = t1 - t0
    onnx_single_times.append(onnx_single_time)

    # Keras single
    t2 = time.perf_counter()
    for i in range(features.shape[0]):
        test_case = features[i].reshape(1, -1, 2)
        _ = keras_model.predict(test_case, verbose=0)

    t3 = time.perf_counter()
    keras_single_time = t3 - t2
    keras_single_times.append(keras_single_time)

    # ONNX batch
    t4 = time.perf_counter()
    _ = onnxruntime_session.run(None, {onnxruntime_session.get_inputs()[0].name: features})
    t5 = time.perf_counter()
    onnx_batch_time = t5 - t4
    onnx_batch_times.append(onnx_batch_time)

    # Keras batch
    t6 = time.perf_counter()
    _ = keras_model.predict(features, verbose=0)
    t7 = time.perf_counter()
    keras_batch_time = t7 - t6
    keras_batch_times.append(keras_batch_time)

    # Save row
    rows.append([
        f"Run {run}",
        round(onnx_single_time, 4),
        round(keras_single_time, 4),
        round(onnx_batch_time, 4),
        round(keras_batch_time, 4)
    ])

# Add average row
rows.append([
    "Average",
    round(np.mean(onnx_single_times), 4),
    round(np.mean(keras_single_times), 4),
    round(np.mean(onnx_batch_times), 4),
    round(np.mean(keras_batch_times), 4)
])

# Write CSV
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(rows)

print(f"Benchmark completed. Results saved to: {csv_filename}")
