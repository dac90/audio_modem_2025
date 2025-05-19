import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
"""
# Parameters
duration = 5  # seconds
samplerate = 44100  # Hz
filename = 'recorded_audio.wav'

# Record audio
print("Recording...")
audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()
print("Recording finished.")

# Save audio to file
write(filename, samplerate, audio)
print(f"Audio saved to {filename}")

# Plot audio
audio_np = audio.flatten()
time = np.linspace(0, duration, len(audio_np))

plt.figure(figsize=(10, 4))
plt.plot(time, audio_np)
plt.title("Recorded Audio Waveform")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
"""


import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.io.wavfile import write

# Settings
duration = 5.0       # seconds
f0 = 500.0           # start frequency
f1 = 2500.0          # end frequency
samplerate = 44100   # Hz
channels = 1
"""
# Generate chirp signal
t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
chirp_signal = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')

# Normalize for output
chirp_signal = chirp_signal.astype(np.float32)

# Buffer to record
recorded = np.empty((int(samplerate * duration), channels), dtype=np.float32)

# Play chirp and record simultaneously
print("Playing chirp and recording...")
sd.play(chirp_signal, samplerate)
recorded = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels)
sd.wait()
print("Recording complete.")

# Save recorded signal
write("recorded_response.wav", samplerate, recorded)

# Plot both signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, chirp_signal)
plt.title("Original Chirp Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(t, recorded.flatten())
plt.title("Recorded Signal (Microphone Input)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
"""

from scipy.signal import csd, welch
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import csd, welch

# Parameters
duration = 3.0  # seconds
samplerate = 44100  # Hz
nperseg = 2048  # for spectral estimation
signal_length = int(duration * samplerate)
# Generate pseudo-random signal (white Gaussian noise)
t = np.linspace(0, duration, signal_length, endpoint=False)
x = chirp(t, f0=500, f1=4000, t1=duration, method='linear')
x = 0.9 * x / np.max(np.abs(x))  # normalize and amplify

# Add silence padding
pad = np.zeros(int(0.5 * samplerate))
x_padded = np.concatenate((pad, x, pad))
print(x)

# Play and record
print("Playing and recording...")
sd.wait()
y = sd.playrec(x_padded, samplerate, channels=1)
sd.wait()
print("Recording complete.")

# Remove padding
x_valid = x
y_valid = y[len(pad):len(pad)+len(x)].flatten()

# Compute cross-spectral and power spectral densities
f, Sxy = csd(y_valid, x_valid, fs=samplerate, nperseg=nperseg)
_, Sxx = welch(x_valid, fs=samplerate, nperseg=nperseg)

# Estimate channel transfer function H(w)
H_est = Sxy / (Sxx)
print("SNR estimate:", 10 * np.log10(np.mean(x_valid**2) / np.mean((y_valid - x_valid)**2)))
# Plot frequency response magnitude
plt.figure(figsize=(10, 5))
plt.semilogy(f, np.abs(H_est))
plt.title("Estimated Channel Frequency Response |H(ω)|")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

