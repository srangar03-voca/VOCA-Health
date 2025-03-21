import os
import librosa
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

# Augmentation functions
def add_noise(y, noise_factor=0.02):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def shift_pitch(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

# Function to extract and resize Mel spectrogram features to a fixed shape
def extract_mel_spectrogram(y, sr=16000, n_mels=128, fixed_shape=(128, 100)):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_resized = cv2.resize(mel_db, fixed_shape, interpolation=cv2.INTER_AREA)
    return mel_resized.flatten()

# Directory containing audio files
audio_dir = r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/mel_class/renamed_audio_files'

# Load first 77 disordered files and 77 normal files
normal_files = []
disorder_files = []

for file_name in os.listdir(audio_dir):
    if file_name.endswith(".wav"):
        if file_name.startswith("N_"):
            normal_files.append(file_name)
        elif file_name.startswith("D_"):
            disorder_files.append(file_name)

selected_normal_files = normal_files[:77]
selected_disorder_files = disorder_files[:77]

# Load, label, and augment data
original_audio_files = []
original_labels = []
augmented_audio_files = []
augmented_labels = []

# Augmentation factor reduced to 2x original data to reduce overfitting
for file_name in selected_normal_files + selected_disorder_files:
    file_path = os.path.join(audio_dir, file_name)
    y, sr = librosa.load(file_path, sr=16000)

    # Label based on filename convention
    label = 0 if file_name.startswith("N_") else 1
    original_audio_files.append(y)
    original_labels.append(label)

    # Add original sample
    augmented_audio_files.append(y)
    augmented_labels.append(label)

    # Apply fewer augmentations (2 per sample)
    augmented_audio_files.append(add_noise(y))
    augmented_labels.append(label)

    augmented_audio_files.append(shift_pitch(y, sr, n_steps=2))
    augmented_labels.append(label)

# Extract Mel spectrograms for augmented data
X_augmented = np.array([extract_mel_spectrogram(y) for y in augmented_audio_files])
y_augmented = np.array(augmented_labels)

# Print data counts before and after augmentation
normal_count = len([label for label in original_labels if label == 0])
disorder_count = len([label for label in original_labels if label == 1])
augmented_normal_count = len([label for label in augmented_labels if label == 0])
augmented_disorder_count = len([label for label in augmented_labels if label == 1])

print(f"Data before augmentation: Normal = {normal_count}, Disorder = {disorder_count}")
print(f"Data after augmentation: Normal = {augmented_normal_count}, Disorder = {augmented_disorder_count}")

# Implement Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
f1_scores = []
confusion_matrices = []
training_results = []

# Lists to collect predictions and true labels from all folds
all_y_true = []
all_y_pred = []
all_y_proba = []

for fold, (train_index, test_index) in enumerate(skf.split(X_augmented, y_augmented)):
    print(f"\nFold {fold + 1}")
    X_train_fold, X_test_fold = X_augmented[train_index], X_augmented[test_index]
    y_train_fold, y_test_fold = y_augmented[train_index], y_augmented[test_index]

    # Normalize features based on training data
    mean = np.mean(X_train_fold, axis=0)
    std = np.std(X_train_fold, axis=0)
    X_train_fold = (X_train_fold - mean) / std
    X_test_fold = (X_test_fold - mean) / std

    # Convert data into DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dtest = xgb.DMatrix(X_test_fold, label=y_test_fold)

    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'error'],
        'eta': 0.1,
        'max_depth': 4,
        'lambda': 1.0,
        'verbosity': 0
    }

    # Evaluation list
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    # Train the model
    num_rounds = 100
    evals_result = {}
    model = xgb.train(
        params, dtrain, num_rounds, evals=evals,
        early_stopping_rounds=10, evals_result=evals_result, verbose_eval=False
    )

    # Record training results
    training_results.append(evals_result)

    # Predict on test data
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Collect predictions and true labels
    all_y_true.extend(y_test_fold)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_pred_proba)

    # Calculate metrics
    accuracy = accuracy_score(y_test_fold, y_pred) * 100
    f1 = f1_score(y_test_fold, y_pred)
    conf_matrix = confusion_matrix(y_test_fold, y_pred)

    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    confusion_matrices.append(conf_matrix)

    print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%")
    print(f"Fold {fold + 1} F1 Score: {f1:.4f}")
    print(f"Fold {fold + 1} Confusion Matrix:\n{conf_matrix}")

# Convert lists to numpy arrays
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_proba = np.array(all_y_proba)

# Compute overall metrics
overall_accuracy = accuracy_score(all_y_true, all_y_pred) * 100
overall_f1 = f1_score(all_y_true, all_y_pred)
overall_conf_matrix = confusion_matrix(all_y_true, all_y_pred)

print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
print(f"Overall F1 Score: {overall_f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=['Normal', 'Disorder']))

# Plot Aggregated Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    overall_conf_matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=['Normal', 'Disorder'], yticklabels=['Normal', 'Disorder']
)
plt.title("Aggregated Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(all_y_true, all_y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(all_y_true, all_y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Plot Training and Validation Metrics for the Last Fold (as an example)
evals_result = training_results[-1]
epochs = len(evals_result['train']['logloss'])
x_axis = range(epochs)

# Plot log loss
plt.figure(figsize=(10, 5))
plt.plot(x_axis, evals_result['train']['logloss'], label='Train')
plt.plot(x_axis, evals_result['eval']['logloss'], label='Validation')
plt.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss - Last Fold')
plt.show()

# Plot classification error
plt.figure(figsize=(10, 5))
plt.plot(x_axis, evals_result['train']['error'], label='Train')
plt.plot(x_axis, evals_result['eval']['error'], label='Validation')
plt.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error - Last Fold')
plt.show()
