import argparse
import time

import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from RoadCharacteristics import ExtractRoadCharacteristics
from DatasetPreprocessing import CombineFiles


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def full_train(road_characteristics):
    angle_data = road_characteristics['segment_angles']
    length_data = road_characteristics['segment_lengths']
    label_data = road_characteristics['labels']

    X_data = []
    for angles, lengths in zip(angle_data, length_data):
        segment_features = np.column_stack((angles, lengths))
        X_data.append(segment_features)

    X_train = np.array(X_data)
    y_train = np.array(label_data)

    t0 = time.time()

    print("Training has started.")

    # Model initialisation

    model = Sequential()
    model.add(Bidirectional(LSTM(units=lstm_cells, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # Save the model
    model.save(os.path.join(model_path, 'model_full.h5'))

    print("Training has finished ", round(time.time() - t0,2), "s")

def train_and_validate(road_characteristics, k_fold):
    angle_data = road_characteristics['segment_angles']
    length_data = road_characteristics['segment_lengths']
    label_data = road_characteristics['labels']

    X_data = []
    for angles, lengths in zip(angle_data, length_data):
        segment_features = np.column_stack((angles, lengths))
        X_data.append(segment_features)

    X_data = np.array(X_data)
    y_data = np.array(label_data)

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    fold_num = 1
    performance_metrics = []

    os.makedirs(os.path.join(model_path, 'confusion_matrices'), exist_ok=True)

    t0 = time.time()

    print("Training has started.")

    for train_index, validation_index in kf.split(X_data):
        X_train, X_validation = X_data[train_index], X_data[validation_index]
        y_train, y_validation = y_data[train_index], y_data[validation_index]

        # model initialization
        model = Sequential()
        model.add(Bidirectional(
            LSTM(units=lstm_cells, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


        # Train the model
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (X_validation, y_validation),\
                  verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)])


        # Predictions and confusion matrices
        y_pred = (model.predict(X_validation) > 0.5).astype('int32')
        cm = confusion_matrix(y_validation, y_pred)

        # Save confusion matrix as image
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot()
        plt.title(f'Confusion Matrix - Fold {fold_num}')
        cm_filename = os.path.join(model_path, 'confusion_matrices', f'confusion_matrix_fold_{fold_num}.png')
        plt.savefig(cm_filename, dpi=900, bbox_inches='tight')
        plt.close()

        # Calculate the performance matrices
        accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
        precision = precision_score(y_validation, y_pred)
        recall = recall_score(y_validation, y_pred)
        f1 = f1_score(y_validation, y_pred)

        performance_metrics.append({
            'fold' : fold_num,
            'accuracy': accuracy,
            'precision': precision,
            'recall' :recall,
            'f1_score': f1,
            'confusion_matrix_file': cm_filename
        })

        print(f"Fold {fold_num} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        _model_file_path = os.path.join(model_path, 'model_files')
        model.save(os.path.join(_model_file_path, f'model_fold_{fold_num}.h5'))

        fold_num += 1

    print("Training has finished. ", round(time.time() - t0, 2), 's')

    metrics_df = pd.DataFrame(performance_metrics)
    metrics_df.to_csv('cross_validation_results.csv', index=False)


def main():

    data_dir = os.path.join(project_root, config.get('data_dir', 'data/'))

    training_mode = config.get('training_mode', 'crossfold')

    combined_dataset_filename = str(config.get('combined_dataset_filename', 'dataset_combined.json'))

    interpolated_road_points_size = config.get('interpolated_road_points_size', 197)

    #CombineFiles(data_dir, combined_dataset_filename, interpolated_road_points_size) # combine the files in the dataset within a single JSON file.

    combined_dataset_filename = os.path.join(data_dir, combined_dataset_filename)

    road_characteristics = ExtractRoadCharacteristics(feature='angles-lengths', combined_dataset_filename=combined_dataset_filename).get_road_characteristics() # extract the road characteristics that will be used for training.

    if training_mode == 'crossvalidate':
        train_and_validate(road_characteristics=road_characteristics, k_fold=k_fold)

    elif training_mode == 'full':
        full_train(road_characteristics=road_characteristics)

    else:
        raise ValueError('Undeclared training mode.')

    # RUN TRAINING


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False,
                        default='experiments/configs/config.yaml',
                        help="Path to experiment config file")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config = load_config(args.config)
    print("Config loaded:", config)

    lstm_cells = config.get('parameters', {}).get('lstm_cells', 220)
    learning_rate = config.get('parameters', {}).get('learning_rate', 1e-3)
    batch_size = config.get('parameters', {}).get('batch_size', 1024)
    epochs = config.get('parameters', {}).get('epochs', 200)

    k_fold = config.get('k_fold', 10)

    model_path = config.get('model_dir', 'data/saved_models')

    main()