import json
import os.path
from datetime import datetime
from RoadCharacteristics import ExtractRoadCharacteristics
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def test_on_trained_data(trained_model, test_file, data_dir):

    print("TESTING...")

    road_characteristics = ExtractRoadCharacteristics(feature='angles-lengths', combined_dataset_filename=test_file).get_road_characteristics()

    angle_data = road_characteristics['segment_angles']
    length_data = road_characteristics['segment_lengths']
    label_data = road_characteristics['labels']

    X_data = []
    for angles, lengths in zip(angle_data, length_data):
        segment_features = np.column_stack((angles, lengths))
        X_data.append(segment_features)

    X_test = np.array(X_data)
    y_test = np.array(label_data)

    print("Trained model:", trained_model)

    model = tf.keras.models.load_model(trained_model)

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    now = datetime.now()

    performance_metrics = [{'Accuracy': acc,
                            'Precision': prec,
                            'Recall': rec,
                            'F1-Score':f1,
                            'Time': now.strftime("%Y-%m-%d %H:%M:%S")}]

    print("Model Performance:")
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1 Score:  {f1:.2f}")

    results_path = os.path.join(data_dir, 'test_evaluation_results.csv')

    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        new_df = pd.DataFrame(performance_metrics)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = pd.DataFrame(performance_metrics)

    updated_df.to_csv(results_path, index=False)

    print("Model results saved to:", results_path)
