import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold                                         
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import warnings
warnings.filterwarnings("ignore")

# Augmentation functions in a class
class AudioAugmentor:
    def __init__(self, noise_factor=0.02, pitch_shift_steps=2, time_stretch_factors=[0.8, 1.2]):
        self.noise_factor = noise_factor
        self.pitch_shift_steps = pitch_shift_steps
        self.time_stretch_factors = time_stretch_factors

    def add_noise(self, y):
        noise = np.random.randn(len(y))
        return y + self.noise_factor * noise

    def shift_pitch(self, y, sr):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=self.pitch_shift_steps)

    def stretch_time(self, y, stretch_factor):
        return librosa.effects.time_stretch(y, rate=stretch_factor)

    def augment(self, y, sr):
        augmented_samples = [y]
        augmented_samples.append(self.add_noise(y))
        augmented_samples.append(self.shift_pitch(y, sr))
        for factor in self.time_stretch_factors:
            augmented_samples.append(self.stretch_time(y, factor))
        return augmented_samples

# Feature extraction functions
def extract_mel_spectrogram(y, sr=16000, n_mels=128, fixed_shape=(128, 100)):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_resized = cv2.resize(mel_db, fixed_shape, interpolation=cv2.INTER_AREA)
    return mel_resized.flatten()

class VoiceDisorderModel:
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.original_audio_files = []
        self.original_labels = []
        self.augmented_audio_files = []
        self.augmented_labels = []
        self.X_augmented = None
        self.y_augmented = None
        self.mean = None
        self.std = None
        self.augmentor = AudioAugmentor()

    def load_and_augment_data(self):
        normal_files, disorder_files = self._get_file_lists()
        selected_normal_files = normal_files[:77]
        selected_disorder_files = disorder_files[:77]

        for file_name in selected_normal_files + selected_disorder_files:
            file_path = os.path.join(self.audio_dir, file_name)
            y, sr = librosa.load(file_path, sr=16000)

            label = 0 if file_name.startswith("N_") else 1
            self.original_audio_files.append(y)
            self.original_labels.append(label)

            # Perform augmentations using the AudioAugmentor class
            augmented_samples = self.augmentor.augment(y, sr)

            # Add original and augmented samples to the dataset
            for sample in augmented_samples:
                self.augmented_audio_files.append(sample)
                self.augmented_labels.append(label)

        # Extract Mel spectrograms for augmented data
        self.X_augmented = np.array([extract_mel_spectrogram(y) for y in self.augmented_audio_files])
        self.y_augmented = np.array(self.augmented_labels)

        self._print_data_counts()

    def _get_file_lists(self):
        normal_files = []
        disorder_files = []
        for file_name in os.listdir(self.audio_dir):
            if file_name.endswith(".wav"):
                if file_name.startswith("N_"):
                    normal_files.append(file_name)
                elif file_name.startswith("D_"):
                    disorder_files.append(file_name)
        return normal_files, disorder_files

    def _print_data_counts(self):
        normal_count = len([label for label in self.original_labels if label == 0])
        disorder_count = len([label for label in self.original_labels if label == 1])
        augmented_normal_count = len([label for label in self.augmented_labels if label == 0])
        augmented_disorder_count = len([label for label in self.augmented_labels if label == 1])

        print(f"Data before augmentation: Normal = {normal_count}, Disorder = {disorder_count}")
        print(f"Data after augmentation: Normal = {augmented_normal_count}, Disorder = {augmented_disorder_count}")

    def train_and_evaluate_model(self):
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_no = 1
        all_f1_scores = []
        all_accuracies = []

        for train_index, test_index in kfold.split(self.X_augmented, self.y_augmented):
            X_train, X_test = self.X_augmented[train_index], self.X_augmented[test_index]
            y_train, y_test = self.y_augmented[train_index], self.y_augmented[test_index]

            # Normalize features
            self.mean = np.mean(X_train, axis=0)
            self.std = np.std(X_train, axis=0)
            X_train = (X_train - self.mean) / self.std
            X_test = (X_test - self.mean) / self.std

            # Build the model
            model = self._build_model(X_train.shape[1])

            # Compile and train the model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
            )

            # Evaluate the model
            self._evaluate_model(model, X_test, y_test, history, fold_no, all_f1_scores, all_accuracies)
            fold_no += 1

        print(f"Average F1 Score across all folds: {np.mean(all_f1_scores):.2f}")
        print(f"Average Accuracy across all folds: {np.mean(all_accuracies):.2f}%")

        # Save the final model and normalization values from the last fold
        model.save('voice_disorder_model_c8.keras')
        np.save('mean_std_values_c8.npy', [self.mean, self.std])

    def _build_model(self, input_shape):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,), kernel_regularizer=tf.keras.regularizers.l2(0.05)),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def _evaluate_model(self, model, X_test, y_test, history, fold_no, all_f1_scores, all_accuracies):
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        # Calculate accuracy using sklearn's accuracy_score
        accuracy = accuracy_score(y_test, y_pred) * 100
        all_accuracies.append(accuracy)

        # Calculate F1 Score
        f1 = f1_score(y_test, y_pred)
        all_f1_scores.append(f1)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Fold {fold_no} - Confusion Matrix:")
        print(conf_matrix)

        # Plot Training and Validation Loss
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

        # Plot Training and Validation Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.show()

if __name__ == "__main__":
    audio_directory = r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/mel_class/renamed_audio_files'
    model = VoiceDisorderModel(audio_directory)
    model.load_and_augment_data()
    model.train_and_evaluate_model()
