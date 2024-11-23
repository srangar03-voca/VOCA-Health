import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment
import soundfile as sf

# Function to remove silence, reduce noise, and normalize volume
def preprocess_audio(audio_file, output_file):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=16000)
    
    # 1. Remove Silence
    y, _ = librosa.effects.trim(y, top_db=20)  # Adjust top_db for silence sensitivity
    
    # 2. Noise Reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)  # Reduces background noise
    
    # 3. Volume Normalization
    # Convert to AudioSegment for normalization with pydub
    y_audio_segment = AudioSegment(
        y_denoised.tobytes(),
        frame_rate=sr,
        sample_width=y_denoised.dtype.itemsize,
        channels=1
    )
    normalized_audio = y_audio_segment.apply_gain(-y_audio_segment.dBFS)  # Normalize to 0 dBFS
    
    # Export the processed audio to a file
    normalized_audio.export(output_file, format="wav")

    # Load back the normalized audio as a numpy array
    y_normalized, _ = librosa.load(output_file, sr=sr)
    
    return y_normalized, sr

# Example usage
audio_file = 'path/to/input_audio.wav'
output_file = 'path/to/processed_audio.wav'
processed_audio, sr = preprocess_audio(audio_file, output_file)
