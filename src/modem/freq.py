import numpy as np
from statsmodels.robust.scale import huber
from scipy.stats import trim_mean
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt
from modem.constants import DATA_BLOCK_LENGTH, DATA_PER_PILOT, LOWER_FREQUENCY_BOUND, UPPER_FREQUENCY_BOUND

def get_freq_gains(observed_freq_gains, plot: bool = True):
    observed_freq_mag = np.abs(observed_freq_gains)
    observed_freq_phase = np.angle(observed_freq_gains)
    pilot_phase = observed_freq_phase[~np.isnan(observed_freq_phase).any(axis=1)]
    d_pilot_phase = pilot_phase[1:,:]-pilot_phase[:-1,:]
    d_pilot_phase[d_pilot_phase < -1.5*np.pi] += 2 * np.pi
    d_pilot_phase[d_pilot_phase > 1.5*np.pi] -= 2 * np.pi

    x = np.arange(np.shape(pilot_phase)[1]).reshape(-1, 1)
    y = trim_mean(d_pilot_phase,  proportiontocut=0.1, axis=0)
    # y = np.array([huber(column)[0] for column in d_pilot_phase.T])

    model = HuberRegressor(epsilon=1.35)  # lower epsilon → more robust (default is 1.35)
    model.fit(x, y)

    pilot_drift = model.predict(x)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', label='Robust d_mean')
        ax.plot(x, pilot_drift, 'r-', label='Huber Regression Fit', linewidth=2)
        ax.set_title("Huber Regression on Robust d_mean")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    freq_phase_drift = pilot_drift / (1 + DATA_PER_PILOT)
    observed_freq_phase_intercept = observed_freq_phase-(np.arange(np.shape(observed_freq_gains)[0]).reshape(-1, 1) * freq_phase_drift)
    observed_freq_intercept = observed_freq_mag * np.exp(1j * observed_freq_phase_intercept)
    freq_intercept = np.nanmean(observed_freq_intercept,axis=0)
    freq_gains = freq_intercept * np.exp(1j * np.arange(np.shape(observed_freq_gains)[0]).reshape(-1, 1) * freq_phase_drift)

    if plot:
        plot_freq_intercept(observed_freq_intercept, 500)

    if plot:
        fig, ax = plt.subplots()
        freqs = np.linspace(LOWER_FREQUENCY_BOUND, UPPER_FREQUENCY_BOUND, DATA_BLOCK_LENGTH, endpoint=False)
        for i, block_gain in enumerate(observed_freq_gains[~np.isnan(observed_freq_gains).any(axis=1)]):
            ax.plot(freqs, np.log10(np.abs(block_gain)), label=f"Block {i}")
        ax.plot(freqs, np.mean(np.log10(np.abs(freq_gains)), axis=0), label="Mean")
        ax.set_title("Frequency Gain Plot (in dB)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.legend()

    return freq_gains

def plot_freq_intercept(observed_freq_intercept, freq):
    # Choose the index j to plot
    j = freq  # example index

    # Collect valid (index, complex value) pairs
    valid_points = [
        (i, arr[j]) for i, arr in enumerate(observed_freq_intercept)
        if arr is not None and j < len(arr)
    ]

    # Separate into components for plotting
    indices = [i for i, _ in valid_points]
    complex_values = [z for _, z in valid_points]
    real_parts = [z.real for z in complex_values]
    imag_parts = [z.imag for z in complex_values]

    # Normalize indices to [0, 1] for coloring
    norm = plt.Normalize(min(indices), max(indices))
    cmap = plt.cm.viridis  # You can try 'plasma', 'cool', etc.
    colors = [cmap(norm(i)) for i in indices]

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)
    scatter = plt.scatter(real_parts, imag_parts, c=colors, cmap=cmap, marker='o')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title(f'Argand Diagram for index j = {j}')
    plt.grid(True)
    plt.axis('equal')

    # Optional: add colorbar to show index mapping
    cbar = plt.colorbar(scatter, label='Index in frequency_gains')
