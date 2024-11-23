import os
import librosa
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import warnings
warnings.filterwarnings("ignore")

# Class for Audio Data Augmentation
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

# Class for Feature Extraction
class FeatureExtractor:
    @staticmethod
    def extract_mel_spectrogram(y, sr=16000, n_mels=128, fixed_shape=(128, 100)):
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_resized = cv2.resize(mel_db, fixed_shape, interpolation=cv2.INTER_AREA)
        return mel_resized.flatten()

# Class for Audio Data Handling
class AudioDataHandler:
    def __init__(self, audio_dir, augmentor, feature_extractor):
        self.audio_dir = audio_dir
        self.augmentor = augmentor
        self.feature_extractor = feature_extractor
        self.audio_data = []
        self.labels = []
        self.normal_count = 0
        self.disorder_count = 0

    def load_and_augment_data(self):
        for file_name in os.listdir(self.audio_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(self.audio_dir, file_name)
                y, sr = librosa.load(file_path, sr=16000)

                # Label based on filename convention
                if file_name.startswith("N_"):
                    label = 0  # Normal voice
                    self.normal_count += 1
                elif file_name.startswith("D1_") or file_name.startswith("D2_") or file_name.startswith("D3_"):
                    label = 1  # Disordered voice
                    self.disorder_count += 1
                else:
                    continue

                # Original audio
                self.audio_data.append(self.feature_extractor.extract_mel_spectrogram(y))
                self.labels.append(label)

                # Apply augmentations using the AudioAugmentor class
                augmented_samples = self.augmentor.augment(y, sr)
                for sample in augmented_samples[1:]: 
                    self.audio_data.append(self.feature_extractor.extract_mel_spectrogram(sample))
                    self.labels.append(label)

    def get_data(self):
        return np.array(self.audio_data), np.array(self.labels)

    def print_data_counts(self):
        print(f"Data before augmentation: Normal = {self.normal_count}, Disorder = {self.disorder_count}")
        augmented_normal_count = len([label for label in self.labels if label == 0])
        augmented_disorder_count = len([label for label in self.labels if label == 1])
        print(f"Data after augmentation: Normal = {augmented_normal_count}, Disorder = {augmented_disorder_count}")

# Class for Model Training and Evaluation
class ModelTrainer:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.mean = None
        self.std = None

    def normalize_features(self, X_train, X_test):
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)
        X_train = (X_train - self.mean) / self.std
        X_test = (X_test - self.mean) / self.std
        return X_train, X_test

    def train_and_evaluate(self):
    
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train, X_test = self.normalize_features(X_train, X_test)

        # K-Fold Cross Validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='f1')

        # Train the model
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

        # Predict and evaluate
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred) * 100

        print(f"Average F1 Score across all folds: {np.mean(scores):.2f}")
        print(f"Overall Model Accuracy (using accuracy_score): {accuracy:.2f}%")

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Disorder'], yticklabels=['Normal', 'Disorder'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        results = self.model.evals_result()

        # Plot Training and Validation Loss
        plt.figure(figsize=(10, 5))
        plt.plot(results['validation_0']['logloss'], label='Training Loss')
        plt.plot(results['validation_1']['logloss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

        # Plot Training and Validation Accuracy
        plt.figure(figsize=(10, 5))
        training_accuracy = [1 - x for x in results['validation_0']['error']]
        validation_accuracy = [1 - x for x in results['validation_1']['error']]
        plt.plot(training_accuracy, label='Training Accuracy')
        plt.plot(validation_accuracy, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.show()

        self.model.save_model('voice_disorder_model_c9.json')
        np.save('mean_std_values_c9.npy', [self.mean, self.std])

def main():
    audio_dir = r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/Perceptual Voice Qualities Database (PVQD)/new_files'

    # Initialize classes
    augmentor = AudioAugmentor()
    feature_extractor = FeatureExtractor()
    data_handler = AudioDataHandler(audio_dir, augmentor, feature_extractor)

    # Load and augment data
    data_handler.load_and_augment_data()
    X, y = data_handler.get_data()
    data_handler.print_data_counts()

    # Initialize model trainer and train model
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss')
    trainer = ModelTrainer(xgb_model, X, y)
    trainer.train_and_evaluate()

if __name__ == "__main__":
    main()
