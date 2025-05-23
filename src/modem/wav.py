import pathlib

import numpy as np
import scipy.io
import sounddevice as sd

from .constants import FS

ROOT_DIR = pathlib.Path(__file__).parent.resolve() / "../.."
WAV_LOCATION = ROOT_DIR / "files"


def read_wav(filename) -> np.ndarray:
    file_path = WAV_LOCATION / filename
    rate, audio = scipy.io.wavfile.read(file_path)
    assert rate == FS, f"Incompatible audio rate {rate} in {file_path}, expected {FS}"

    return audio


def generate_wav(filename: str, audio: np.ndarray) -> None:
    """Save audio data to filename in WAV_LOCATION"""
    file_path = WAV_LOCATION / filename
    scipy.io.wavfile.write(file_path, FS, audio)
    print(f"Saved to {file_path}")


def record_wav(filename: str, duration: float = 10) -> None:
    """Record audio for duration and save to filename in WAV_LOCATION"""
    print(f"Recording for {duration} seconds at {FS} Hz...")

    audio_data = sd.rec(int(duration * FS), samplerate=FS, channels=1, blocking=True)

    generate_wav(filename, audio_data)


def record_entrypoint():
    """CLI entry point to record audio for 10 seconds"""
    record_wav("output.wav", 10)
