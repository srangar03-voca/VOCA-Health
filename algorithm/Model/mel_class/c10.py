import os
import librosa
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import warnings
import xgboost as xgb

warnings.filterwarnings("ignore")

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

print("\n### Data Summary ###")
print(f"Data Before Augmentation: Normal = {normal_count}, Disorder = {disorder_count}")
print(f"Data After Augmentation: Normal = {augmented_normal_count}, Disorder = {augmented_disorder_count}")

# Split the dataset into training+validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_augmented, y_augmented, test_size=0.2, stratify=y_augmented, random_state=42
)

print(f"Training+Validation Set: {len(X_train_val)} samples")
print(f"Test Set: {len(X_test)} samples")

# Implement Stratified K-Fold Cross-Validation on training+validation set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
f1_scores = []
confusion_matrices = []
training_results = []

# Lists to collect predictions and true labels from all folds
all_y_true = []
all_y_pred = []
all_y_proba = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train_val, y_train_val)):
    print(f"\nFold {fold + 1}")
    X_train_fold, X_val_fold = X_train_val[train_index], X_train_val[val_index]
    y_train_fold, y_val_fold = y_train_val[train_index], y_train_val[val_index]

    # Normalize features within the fold
    mean = np.mean(X_train_fold, axis=0)
    std = np.std(X_train_fold, axis=0)
    X_train_fold = (X_train_fold - mean) / std
    X_val_fold = (X_val_fold - mean) / std

    # Convert data into DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'error'],
        'eta': 0.1,
        'max_depth': 4,
        'lambda': 1.0,
        'verbosity': 0
    }

    # Train the model
    num_rounds = 100
    evals_result = {}
    model = xgb.train(
        params, dtrain, num_rounds, evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=10, evals_result=evals_result, verbose_eval=False
    )

    # Record training results
    training_results.append(evals_result)

    # Predict on validation set
    y_val_pred_proba = model.predict(dval)
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)

    # Collect predictions and true labels for validation
    all_y_true.extend(y_val_fold)
    all_y_pred.extend(y_val_pred)
    all_y_proba.extend(y_val_pred_proba)

    # Calculate metrics for the fold
    accuracy = accuracy_score(y_val_fold, y_val_pred) * 100
    f1 = f1_score(y_val_fold, y_val_pred)
    conf_matrix = confusion_matrix(y_val_fold, y_val_pred)

    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    confusion_matrices.append(conf_matrix)

    print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%")
    print(f"Fold {fold + 1} F1 Score: {f1:.4f}")

# Plot Aggregated Confusion Matrix for Validation Set
overall_conf_matrix = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    overall_conf_matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=['Normal', 'Disorder'], yticklabels=['Normal', 'Disorder']
)
plt.title("Aggregated Confusion Matrix - Validation Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Train final model on the entire training+validation set
X_train_val_normalized = (X_train_val - np.mean(X_train_val, axis=0)) / np.std(X_train_val, axis=0)
dtrain_final = xgb.DMatrix(X_train_val_normalized, label=y_train_val)
final_model = xgb.train(params, dtrain_final, num_rounds)

# Evaluate on the test set
X_test_normalized = (X_test - np.mean(X_train_val, axis=0)) / np.std(X_train_val, axis=0)
dtest = xgb.DMatrix(X_test_normalized)
y_test_pred_proba = final_model.predict(dtest)
y_test_pred = (y_test_pred_proba > 0.5).astype(int)

# Compute metrics on the test set
test_accuracy = accuracy_score(y_test, y_test_pred) * 100
test_f1 = f1_score(y_test, y_test_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

print(f"\n### Test Set Results ###")
print(f"Test Set Accuracy: {test_accuracy:.2f}%")
print(f"Test Set F1 Score: {test_f1:.4f}")
print(f"Test Set Confusion Matrix:\n{test_conf_matrix}")

# Plot Confusion Matrix for Test Set
plt.figure(figsize=(6, 4))
sns.heatmap(
    test_conf_matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=['Normal', 'Disorder'], yticklabels=['Normal', 'Disorder']
)
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot ROC Curve for Test Set
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Set')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve for Test Set
precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Test Set')
plt.legend(loc="lower left")
plt.show()
