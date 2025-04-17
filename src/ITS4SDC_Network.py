import os
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from src.RoadCharacteristics import ExtractRoadCharacteristics

class Network:
    def __init__(self, road_characteristics, config, trained_model_file, test_file):

        self.trained_model_file = trained_model_file
        self.test_file = test_file

        self.lstm_cells = config.get('parameters', {}).get('lstm_cells', 220)
        self.lr = config.get('parameters', {}).get('learning_rate', 1e-3)
        self.batch_size = config.get('parameters', {}).get('batch_size', 1024)
        self.epochs = config.get('parameters', {}).get('epochs', 200)
        self.kf = KFold(n_splits=config.get('k_fold', 10), shuffle=True, random_state=42)
        self.mode = config.get('parameters', {}).get('training_mode', 'cv')
        self.angle_data = road_characteristics['segment_angles']
        self.length_data = road_characteristics['segment_lengths']
        self.label_data = road_characteristics['labels']
        self.features, self.labels = [], []
        self.build_feature_matrix()

        # RE-DEFINE
        self.model_path = config.get('model_dir', 'data/')
        self.data_dir = config.get('data_dir', 'data/')
        self.now = str(datetime.now().strftime("%Y_%m_%d_%H_%M"))
        model_path = os.path.join(self.model_path, 'saved_model_' + self.now)
        confusion_matrix_path = os.path.join(model_path, 'confusion_matrices')
        os.makedirs(confusion_matrix_path, exist_ok=True)

    def build_feature_matrix(self):

        features = []
        for angles, lengths in zip(self.angle_data, self.length_data):
            segment_features = np.column_stack((angles, lengths))
            features.append(segment_features)

        self.features = np.array(features)
        self.labels = np.array(self.label_data)

    def calculate_prediction(self):

        road_characteristics_test_file = ExtractRoadCharacteristics(feature='angles-lengths', combined_dataset_filename=self.test_file).get_road_characteristics()
        angle_data = road_characteristics_test_file['segment_angles']
        length_data = road_characteristics_test_file['segment_lengths']
        label_data = road_characteristics_test_file['labels']

        test_features = []
        for angles, lengths in zip(angle_data, length_data):
            segment_features = np.column_stack((angles, lengths))
            test_features.append(segment_features)

        test_features = np.array(test_features)
        test_labels = np.array(label_data)

        # load the trained model
        model = tf.keras.models.load_model(self.trained_model_file)
        y_pred = model.predict(test_features)
        predicted_labels = (y_pred > 0.5).astype("int32")

        acc = accuracy_score(test_labels, predicted_labels)
        prec = precision_score(test_labels, predicted_labels)
        rec = recall_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels)

        now = datetime.now()

        performance_metrics = [{'Accuracy': acc,
                                'Precision': prec,
                                'Recall': rec,
                                'F1-Score': f1,
                                'Time': now.strftime("%Y-%m-%d %H:%M:%S")}]

        print("Model Performance:")
        print(f"Accuracy:  {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall:    {rec:.2f}")
        print(f"F1 Score:  {f1:.2f}")
        results_path = os.path.join(self.data_dir, 'test_evaluation_results.csv')

        if os.path.exists(results_path):
            existing_df = pd.read_csv(results_path)
            new_df = pd.DataFrame(performance_metrics)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = pd.DataFrame(performance_metrics)

        updated_df.to_csv(results_path, index=False, sep=';')

        print("Model results saved to:", results_path)

    def model_pipeline(self):

        if self.trained_model_file is not None: # if model file is provided; skip training and do testing.
            self.calculate_prediction()

        # Build the model
        model = Sequential()
        model.add(Bidirectional(LSTM(units=self.lstm_cells, input_shape=(self.features.shape[1], self.features.shape[2]), return_sequences=False)))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        mode = self.mode.lower()

        if mode == 'cross-validation':
            kf = KFold(n_splits=self.kf, shuffle=True, random_state=42)
            fold_num = 1
            performance_metrics = []

            for train_index, validation_index in kf.split(self.features):
                train_features, validation_features = self.features[train_index], self.features[validation_index]
                train_labels, validation_labels = self.labels[train_index], self.labels[validation_index]

                model.fit(train_features, train_labels, batch_size=self.batch_size, epochs=self.epochs, validation_data=(validation_features, validation_labels), verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)])

                performance_metrics = self.evaluate_fold_performance(fold_number=fold_num, model=model, X_validation = validation_features, y_validation=validation_labels, performance_metrics=performance_metrics)

                fold_num += 1

            self.export_cv_metrics_to_csv(performance_metrics=performance_metrics)

        elif mode == 'full':
            model.fit(self.features, self.labels, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
            self.export_trained_model(model)

        else:
            raise ValueError('Undeclared mode.')

    def export_cv_metrics_to_csv(self, performance_metrics):

        metrics_df = pd.DataFrame(performance_metrics)
        metrics_df.to_csv(self.data_dir + '/cross_validation_results.csv', index=False)

        best_folds_model_path = max(performance_metrics, key=lambda x: x['f1_score'])['confusion_matrix_file'].replace('confusion_matrices', 'model_files').replace('confusion_matrix_', 'model_fold_').replace('.png', '.keras')

    def evaluate_fold_performance(self, fold_number, model, validation_features, validation_labels, performance_metrics):

        y_pred = (model.predict(validation_features) > 0.5).astype('int32')

        cm = confusion_matrix(validation_labels, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot()
        plt.title(f'Confusion Matrix - Fold {fold_number}')
        cm_filename = os.path.join(self.model_path, 'confusion_matrices', f'confusion_matrix_fold_{fold_number}.png')
        plt.savefig(cm_filename, dpi=900, bbox_inches='tight')
        plt.close()
        # Calculate the performance matrices
        accuracy = accuracy_score(validation_labels, y_pred)
        precision = precision_score(validation_labels, y_pred)
        recall = recall_score(validation_labels, y_pred)
        f1 = f1_score(validation_labels, y_pred)

        performance_metrics.append({
            'fold': fold_number,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix_file': cm_filename
        })

        print(f"Fold {fold_number} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        model.save(os.path.join(self.model_path, 'model_files', f'model_fold_{fold_number}.keras'))

        return performance_metrics

    def export_trained_model(self, model):

        model_save_file = os.path.join(self.model_path, 'saved_model_' + self.now, 'model_full.keras')

        model.save(model_save_file)