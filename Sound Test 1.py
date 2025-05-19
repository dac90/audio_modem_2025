from scipy.signal import csd, welch, chirp
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

