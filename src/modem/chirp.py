import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal

from modem.constants import FS, OFDM_SYMBOL_LENGTH

CHIRP_DURATION = 0.5
CHIRP_LENGTH = int(CHIRP_DURATION * FS)
CHIRP_F0 = 0
CHIRP_F1 = 3000


def generate_chirp() -> npt.NDArray[np.float64]:
    t = np.linspace(0, CHIRP_DURATION, CHIRP_LENGTH)
    chirp = np.sin(np.pi * (CHIRP_F0 + (CHIRP_F1 - CHIRP_F0) * t / CHIRP_DURATION) * t)
    return chirp


# Use reversed chirp for start
START_CHIRP = generate_chirp()[::-1]

# Use forward chirp for end
END_CHIRP = generate_chirp()


def synchronise(recv_signal: npt.NDArray[np.float64], plot_correlations: bool = False,
                plot_spectrogram: bool = False, delay: int = 0):
    """Synchronise received signal assuming a whole number of OFDM_SYMBOL_LENGTH between start and end chirps.
    Returns aligned signal with start and end chirp included.
    
    If plot_correlations, will create a plot with the correlations of the signal with the start and end chirp
    around their maxima.
    
    If a delay is given, it will delay the aligned signal by that many samples (may be negative)."""
    recv_signal = recv_signal.flatten() ###
    start_correlation = scipy.signal.correlate(recv_signal, START_CHIRP, mode="valid")
    lags = scipy.signal.correlation_lags(recv_signal.size, START_CHIRP.size, mode="valid")
    start_lag = lags[np.argmax(np.abs(start_correlation))] - delay

    end_correlation = scipy.signal.correlate(recv_signal, END_CHIRP, mode="valid")
    lags = scipy.signal.correlation_lags(recv_signal.size, END_CHIRP.size, mode="valid")
    end_lag = lags[np.argmax(np.abs(end_correlation))] - delay

    difference = end_lag - start_lag
    number_of_blocks = round((difference - len(START_CHIRP)) / OFDM_SYMBOL_LENGTH)
    expected_difference = number_of_blocks * OFDM_SYMBOL_LENGTH + len(START_CHIRP)
    error = difference - expected_difference
    print(f"Assumed {number_of_blocks} OFDM blocks sent")
    print(f"Synchronisation was {error} samples too long")

    if plot_correlations:
        fig, ax = plt.subplots()
        prefix, suffix = 1000, 1000
        time_vals = np.arange(-prefix, suffix) / FS
        ax.plot(
            time_vals, start_correlation[start_lag - prefix : start_lag + suffix], label="Correlation with start chirp"
        )
        ax.plot(time_vals, end_correlation[end_lag - prefix : end_lag + suffix], label="Correlation with end chirp")
        ax.legend()
        ax.set_xlabel("Time (seconds)")

    aligned_recv_signal = np.roll(recv_signal, -start_lag)[: expected_difference +  END_CHIRP.size]

    if plot_spectrogram:
        fig, ax = plt.subplots()

        f, t_spec, Sxx = scipy.signal.spectrogram(aligned_recv_signal, FS)

        pcm = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading="gouraud")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [sec]")
        cbar = fig.colorbar(pcm, ax=ax, label="Power/Frequency (dB/Hz)")

    return aligned_recv_signal
