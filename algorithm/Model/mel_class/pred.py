import numpy as np
import tensorflow as tf
import librosa
import cv2

# Load the saved model and normalization values
model = tf.keras.models.load_model('voice_disorder_model_c3.h5')
mean, std = np.load('mean_std_values.npy')

# Function to preprocess a single audio file
def preprocess_audio_file(file_path, sr=16000, n_mels=128, fixed_shape=(128, 100), mean=None, std=None):
    # Load and trim silence from the audio file
    y, sr = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Resize the Mel spectrogram to a fixed shape for consistent input size
    mel_resized = cv2.resize(mel_db, fixed_shape, interpolation=cv2.INTER_AREA)
    mel_flattened = mel_resized.flatten()

    # Normalize using the mean and std from training
    if mean is not None and std is not None:
        mel_flattened = (mel_flattened - mean) / std
    
    return mel_flattened.reshape(1, -1)  # Reshape for model input

# Prediction function for a single file to get specific probability
def predict_class_with_probability(file_path, threshold=0.5):
    # Preprocess the audio file
    features = preprocess_audio_file(file_path, mean=mean, std=std)
    
    # Get the probability prediction from the model
    prediction = model.predict(features)
    disorder_probability = prediction[0][0]  # Probability of being disordered

    # Determine the predicted class based on threshold
    if disorder_probability > threshold:
        # Predicted as Disordered
        print(f"File: {file_path}")
        print("Predicted: Disordered")
        print(f"Probability of being Disordered: {disorder_probability * 100:.2f}%")
    else:
        # Predicted as Normal
        normal_probability = 1 - disorder_probability  # Probability of being normal
        print(f"File: {file_path}")
        print("Predicted: Normal")
        print(f"Probability of being Normal: {normal_probability * 100:.2f}%")

# Example usage
# Path to the single audio file
single_audio_file = r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/Perceptual Voice Qualities Database (PVQD)/Audio_Files/PT130_ENSS.wav'  

# Predict and display probability for the predicted class
predict_class_with_probability(single_audio_file)
