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
import onnxruntime
import shutil

class Network:
    def __init__(self, road_characteristics, config, project_root, use_existing_model):
        """
        :param road_characteristics:
        :param config: config.yaml file path.
        :param trained_model_file: will be used for the prediction, if it exists.
        :param test_file: will be used for prediction.
        :param project_root: the path of the root folder.
        :param use_existing_model: if use_existing_model true, the tool directly performs prediction.
        """
        self.project_root = project_root
        self.use_existing_model = use_existing_model
        self.lstm_cells = config.get('parameters', {}).get('lstm_cells', 220)
        self.lr = config.get('parameters', {}).get('learning_rate', 1e-3)
        self.batch_size = config.get('parameters', {}).get('batch_size', 1024)
        self.epochs = config.get('parameters', {}).get('epochs', 200)
        self.kf = KFold(n_splits=config.get('k_fold', 5), shuffle=True, random_state=42)
        self.mode = config.get('parameters', {}).get('training_mode', 'crossvalidate')
        self.angle_data = road_characteristics['segment_angles']
        self.length_data = road_characteristics['segment_lengths']
        self.label_data = road_characteristics['labels']
        self.test_ids = road_characteristics['test_id']
        self.features, self.labels = [], []
        self.build_feature_matrix()
        self.existing_model_path = os.path.join(project_root, config.get('onnx_model_file', ''))

        # RE-DEFINE
        self.now = str(datetime.now().strftime("%Y_%m_%d_%H_%M"))

        self.results_dir = os.path.join(self.project_root, config.get('results_dir', 'data/results/'))

        self.model_path = os.path.join(self.results_dir, 'result_' + self.now)

        os.makedirs(self.model_path, exist_ok=True)

    def build_feature_matrix(self):

        features = []
        for angles, lengths in zip(self.angle_data, self.length_data):
            segment_features = np.column_stack((angles, lengths))
            features.append(segment_features)

        self.features = np.array(features)
        self.labels = np.array(self.label_data)

    def build_model(self):
        # Build the model
        model = Sequential()
        model.add(Bidirectional(
            LSTM(units=self.lstm_cells, input_shape=(self.features.shape[1], self.features.shape[2]),
                 return_sequences=False)))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def model_pipeline(self):

        mode = self.mode.lower()

        if mode == 'crossvalidate':

            fold_num = 1
            performance_metrics = []

            for train_index, validation_index in self.kf.split(self.features):
                train_features, validation_features = self.features[train_index], self.features[validation_index]
                train_labels, validation_labels = self.labels[train_index], self.labels[validation_index]

                model = self.build_model()
                model.fit(train_features, train_labels, batch_size=self.batch_size, epochs=self.epochs, validation_data=(validation_features, validation_labels), verbose=1, callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)])

                self.evaluate_fold_performance(fold_number=fold_num, model=model, validation_features = validation_features, validation_labels = validation_labels, performance_metrics=performance_metrics)

                fold_num += 1

            best_model_path = self.export_cv_metrics_to_csv(performance_metrics=performance_metrics)

            self.deploy_best_model(best_model_path)

        elif mode == 'full':

            model = self.build_model()
            model.fit(self.features, self.labels, batch_size=self.batch_size, epochs=self.epochs, verbose=1)

            model_filename = 'model_full.keras'

            self.export_trained_model(model, os.path.join(self.model_path, model_filename))

        else:
            raise ValueError('Undeclared mode.')

    def deploy_best_model(self, best_model_path):
        shutil.copy(best_model_path, os.path.join(self.project_root, 'data', 'existing_models', 'its4sdc_model.keras'))

    def export_cv_metrics_to_csv(self, performance_metrics):

        metrics_df = pd.DataFrame(performance_metrics)
        results_path = os.path.join(self.model_path, 'cross_validation_results.csv')

        metrics_df.to_csv(results_path, index=False)

        best_fold = max(performance_metrics, key=lambda x: x['f1_score'])['fold']
        best_folds_model_path = os.path.join(self.model_path, 'model_files', f'model_fold_{best_fold}.keras')

        return best_folds_model_path

    def evaluate_fold_performance(self, fold_number, model, validation_features, validation_labels, performance_metrics):

        y_pred = (model.predict(validation_features) > 0.5).astype('int32')

        cm = confusion_matrix(validation_labels, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot()
        plt.title(f'Confusion Matrix - Fold {fold_number}')

        os.makedirs(os.path.join(self.model_path, 'confusion_matrices'), exist_ok=True)

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

        self.export_trained_model(model=model, model_filename=os.path.join(self.model_path, 'model_files', f'model_fold_{fold_number}.keras'))

        return performance_metrics

    def export_trained_model(self, model, model_filename):

        os.makedirs(os.path.join(self.model_path, 'model_files'), exist_ok=True)

        model.save(model_filename)

    def run_prediction(self):

        predicted_test_outcomes = []

        if self.existing_model_path.endswith('.onnx') and os.path.exists(self.existing_model_path):
            session = onnxruntime.InferenceSession(self.existing_model_path)
            input_name = session.get_inputs()[0].name
            for i in range(self.features.shape[0]):
                test_case_feature = self.features[i].astype(np.float32).reshape(1, -1, 2)
                prediction = session.run(None, {input_name: test_case_feature})
                score = prediction[0][0][0]
                predicted_test_outcomes.append(0 if score < 0.5 else 1)

        else:
            from pathlib import Path

            model_path = Path(self.existing_model_path)

            keras_model_path = model_path.parent / 'its4sdc_model.keras'

            model_ = tf.keras.models.load_model(str(keras_model_path))

            for i in range(self.features.shape[0]):
                test_case_feature = self.features[i].astype(np.float32).reshape(1, -1, 2)
                prediction = model_.predict(test_case_feature, verbose=0)
                score = np.squeeze(prediction)
                predicted_test_outcomes.append(0 if score < 0.5 else 1)


        # Write the predicted outcomes to a CSV file
        predicted_test_outcomes = np.array(predicted_test_outcomes)
        output_filename = os.path.join(self.model_path, 'predicted_outcomes.csv')
        df = pd.DataFrame({'Actual': self.labels, 'Predicted Test Outcomes': predicted_test_outcomes, 'Test Id': self.test_ids})
        df.to_csv(output_filename, index=False)
        print("[INFO] Predicted outcomes saved to:", output_filename)