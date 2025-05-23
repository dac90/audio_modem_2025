import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os

# Parameters for recording
DURATION = 5  # Duration of the recording in seconds
FS = 44100    # Sampling frequency (samples per second)

# Function to record audio
def record_audio(duration, fs):
    print("Recording...")
    audio = sd.rec(int((duration+10) * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
    return audio

# Function to save audio to a WAV file
def save_audio(filename, audio, fs):
    wav.write(filename, fs, audio)
    print(f"Audio saved to {filename}")

# Main program
if __name__ == "__main__":
    output_file = "outputchirp.wav"  # Name of the output WAV file
    recorded_audio = record_audio(DURATION, FS)
    save_audio(output_file, recorded_audio, FS)
    print("Current working directory:", os.getcwd())