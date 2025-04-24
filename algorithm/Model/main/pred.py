import librosa
import numpy as np
import xgboost as xgb
import cv2


class VoiceDisorderPredictor:
    def __init__(self, model_path, mean_std_path):
        """
        Initializes the predictor by loading the trained model and normalization values.
        """
        # Load the trained XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Load the mean and standard deviation for normalization
        self.mean, self.std = np.load(mean_std_path)

    @staticmethod
    def extract_mel_spectrogram(y, sr=16000, n_mels=128, fixed_shape=(128, 100)):
        """
        Extracts and resizes a Mel spectrogram from the audio signal.
        """
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_resized = cv2.resize(mel_db, fixed_shape, interpolation=cv2.INTER_AREA)
        return mel_resized.flatten()

    @staticmethod
    def extract_audio_features(y, sr):
        """
        Extracts fundamental frequency (F0), pitch, tone, and volume.
        """
        # Fundamental frequency (F0) estimation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Tone estimation (Average spectral centroid)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        tone = np.mean(spectral_centroid)
        
        # Volume estimation (Root Mean Square Energy)
        rms = librosa.feature.rms(y=y)
        volume = np.mean(rms)
        
        return {
            "fundamental_frequency": round(pitch, 2),
            "tone": round(tone, 2),
            "volume": round(volume, 2)
        }

    def preprocess_audio(self, file_path):
        """
        Loads and preprocesses an audio file for prediction.
        """
        try:
            y, sr = librosa.load(file_path, sr=16000)
            features = self.extract_mel_spectrogram(y, sr)
            audio_features = self.extract_audio_features(y, sr)

            # Normalize the extracted features
            features_normalized = (features - self.mean) / self.std
            return features_normalized, audio_features
        except Exception as e:
            raise ValueError(f"Error processing file {file_path}: {e}")

    def predict(self, file_path):
        """
        Predicts whether a voice disorder is present or absent for a given audio file.
        """
        # Preprocess the audio file
        features, audio_features = self.preprocess_audio(file_path)
        dmatrix = xgb.DMatrix(np.array([features]))
        
        # Get prediction probabilities
        prob_disorder = self.model.predict(dmatrix)[0]
        prob_normal = 1 - prob_disorder
        label = "Disorder" if prob_disorder > 0.5 else "Normal"

        # Return results with extracted features
        return {
            "prediction": label,
            "probability_normal": round(prob_normal * 100, 2),
            "probability_disorder": round(prob_disorder * 100, 2),
            "fundamental_frequency": audio_features["fundamental_frequency"],
            "tone": audio_features["tone"],
            "volume": audio_features["volume"]
        }


def main():
    # Paths to model and normalization files
    model_path = "/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/mel_class/voice_disorder_model_c10_structured.json"
    mean_std_path = "mean_std_values_c10.npy"  # Ensure this file is in the same directory

    # Initialize the predictor
    predictor = VoiceDisorderPredictor(model_path, mean_std_path)

    # Input the path to the test audio file
    test_file = input(r"/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/Perceptual Voice Qualities Database (PVQD)/new_files/D3_F_4.wav ")

    # Run the prediction
    try:
        result = predictor.predict(test_file)

        # Print the results (without file name)
        print("\n=== Voice Disorder Prediction Results ===")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability of Normal: {result['probability_normal']}%")
        print(f"Probability of Disorder: {result['probability_disorder']}%")
        print(f"Fundamental Frequency (F0): {result['fundamental_frequency']} Hz")
        print(f"Tone (Spectral Centroid): {result['tone']}")
        print(f"Volume (RMS Energy): {result['volume']}\n")
    
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
