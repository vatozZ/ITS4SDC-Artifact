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

class Network:
    def __init__(self, road_characteristics, config, trained_model_file, test_file, project_root):

        self.trained_model_file = trained_model_file
        self.test_file = test_file
        self.project_root = project_root
        self.lstm_cells = config.get('parameters', {}).get('lstm_cells', 220)
        self.lr = config.get('parameters', {}).get('learning_rate', 1e-3)
        self.batch_size = config.get('parameters', {}).get('batch_size', 1024)
        self.epochs = config.get('parameters', {}).get('epochs', 200)
        self.kf = KFold(n_splits=config.get('k_fold', 5), shuffle=True, random_state=42)
        self.mode = config.get('parameters', {}).get('training_mode', 'crossvalidate')
        self.angle_data = road_characteristics['segment_angles']
        self.length_data = road_characteristics['segment_lengths']
        self.label_data = road_characteristics['labels']
        self.features, self.labels = [], []
        self.build_feature_matrix()
        self.onnx_model_path = os.path.join(project_root, config.get('onnx_model_file', ''))

        # RE-DEFINE
        self.now = str(datetime.now().strftime("%Y_%m_%d_%H_%M"))

        self.results_dir = os.path.join(self.project_root, config.get('results_dir', 'data/results/'))

        self.model_path = os.path.join(self.results_dir, 'result_' + self.now)

        confusion_matrix_path = os.path.join(self.model_path, 'confusion_matrices')

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
        results_path = os.path.join(self.results_dir, 'test_evaluation_results.csv')

        if os.path.exists(results_path):
            existing_df = pd.read_csv(results_path)
            new_df = pd.DataFrame(performance_metrics)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = pd.DataFrame(performance_metrics)

        updated_df.to_csv(results_path, index=False)

        print("Model results saved to:", results_path)

    def model_pipeline(self):

        if self.trained_model_file is not None: # if model file is provided; skip training and do testing.
            self.calculate_prediction()
            return

        # Build the model
        model = Sequential()
        model.add(Bidirectional(LSTM(units=self.lstm_cells, input_shape=(self.features.shape[1], self.features.shape[2]), return_sequences=False)))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        mode = self.mode.lower()

        if mode == 'crossvalidate':

            fold_num = 1
            performance_metrics = []

            for train_index, validation_index in self.kf.split(self.features):
                train_features, validation_features = self.features[train_index], self.features[validation_index]
                train_labels, validation_labels = self.labels[train_index], self.labels[validation_index]

                model.fit(train_features, train_labels, batch_size=self.batch_size, epochs=self.epochs, validation_data=(validation_features, validation_labels), verbose=1, callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)])

                self.evaluate_fold_performance(fold_number=fold_num, model=model, validation_features = validation_features, validation_labels = validation_labels, performance_metrics=performance_metrics)

                fold_num += 1

            self.export_cv_metrics_to_csv(performance_metrics=performance_metrics)

        elif mode == 'full':
            model.fit(self.features, self.labels, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
            self.export_trained_model(model)

        else:
            raise ValueError('Undeclared mode.')

    def export_cv_metrics_to_csv(self, performance_metrics):

        metrics_df = pd.DataFrame(performance_metrics)
        results_path = os.path.join(self.model_path, 'cross_validation_results.csv')

        metrics_df.to_csv(results_path, index=False)

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

        os.makedirs(os.path.join(self.model_path, 'model_files'), exist_ok=True)
        model.save(os.path.join(self.model_path, 'model_files', f'model_fold_{fold_number}.keras'))

        return performance_metrics

    def export_trained_model(self, model):


        model_save_file = os.path.join(self.model_path, 'model_full.keras')

        os.makedirs(os.path.dirname(model_save_file), exist_ok=True)

        model.save(model_save_file)

    def run_onnx_prediction(self):

        onnxruntime_session = onnxruntime.InferenceSession(self.onnx_model_path)
        #feature_input_data = np.column_stack((segment_angles, segment_lengths)).astype(np.float32)
        predicted_test_outcomes = []
        for i in range(self.features.shape[0]):
            test_case_feature = self.features[i].astype(np.float32)
            test_case_label = self.labels[i]
            test_case_feature = test_case_feature.reshape(1, -1, 2)
            prediction = onnxruntime_session.run(None, {onnxruntime_session.get_inputs()[0].name: test_case_feature})

            if prediction[0][0][0] < 0.5:
                predicted_test_outcomes.append(0)
            else:
                predicted_test_outcomes.append(1)

        predicted_test_outcomes = np.array(predicted_test_outcomes)
        cm = confusion_matrix(self.labels, predicted_test_outcomes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot()
        plt.title(f'Confusion Matrix')
        cm_filename = os.path.join(self.model_path, 'confusion_matrices', f'confusion_matrix.png')
        plt.savefig(cm_filename, dpi=900, bbox_inches='tight')
        plt.close()

        # Calculate the performance matrices
        accuracy = accuracy_score(self.labels, predicted_test_outcomes)
        precision = precision_score(self.labels, predicted_test_outcomes)
        recall = recall_score(self.labels, predicted_test_outcomes)
        f1 = f1_score(self.labels, predicted_test_outcomes)
        performance_metrics = []
        performance_metrics.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix_file': cm_filename
        })

        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        metrics_df = pd.DataFrame(performance_metrics)
        results_path = os.path.join(self.model_path, 'performance_metric_results.csv')
        metrics_df.to_csv(results_path, index=False)