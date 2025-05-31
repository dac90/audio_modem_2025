import numpy as np
import scipy.io.wavfile
from pathlib import Path

# Constants
FS = 44100  # Sample rate (Hz)
DURATION = 10  # Duration in seconds
WAV_LOCATION = Path("output_wavs")  # Output directory
WAV_LOCATION.mkdir(exist_ok=True)

def generate_wav(filename: str, audio: np.ndarray) -> None:
    """Save audio data (converted to float32) to filename in WAV_LOCATION"""
    file_path = WAV_LOCATION / filename
    audio = audio.astype(np.float32)
    scipy.io.wavfile.write(file_path, FS, audio)
    print(f"Saved to {file_path}")

def generate_white_noise(duration: float, fs: int) -> np.ndarray:
    """Generate white noise with values in the range [-1.0, 1.0]"""
    samples = int(duration * fs)
    noise = np.random.uniform(low=-1.0, high=1.0, size=samples)
    return noise.astype(np.float32)

# Generate and save white noise
white_noise = generate_white_noise(DURATION, FS)
generate_wav("white_noise.wav", white_noise)