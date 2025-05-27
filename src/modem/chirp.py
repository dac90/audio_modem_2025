import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal

from .constants import FS

CHIRP_DURATION = 1.0
CHIRP_F0 = 500
CHIRP_F1 = 4000

CHIRP_TIMES = np.linspace(0, CHIRP_DURATION, int(CHIRP_DURATION * FS), endpoint=False)

# Use upward chirp for start
START_CHIRP = scipy.signal.chirp(CHIRP_TIMES, f0=CHIRP_F0, f1=CHIRP_F1, t1=CHIRP_DURATION, method="linear")

# Use downward chirp for end
END_CHIRP = scipy.signal.chirp(CHIRP_TIMES, f0=CHIRP_F1, f1=CHIRP_F0, t1=CHIRP_DURATION, method="linear")


def synchronise(recv_signal: npt.NDArray[np.complex64], expected_signal_len: int, plot_correlations: bool = False):
    """Synchronise received signal given expected_signal_len between start and end chirps.
    Returns aligned signal with start and end chirp included."""
    start_correlation = scipy.signal.correlate(recv_signal, START_CHIRP, mode="valid")
    lags = scipy.signal.correlation_lags(recv_signal.size, START_CHIRP.size, mode="valid")
    start_lag = lags[np.argmax(np.abs(start_correlation))]

    end_correlation = scipy.signal.correlate(recv_signal, END_CHIRP, mode="valid")
    lags = scipy.signal.correlation_lags(recv_signal.size, END_CHIRP.size, mode="valid")
    end_lag = lags[np.argmax(np.abs(end_correlation))]

    difference = end_lag - start_lag
    expected_difference = expected_signal_len + len(END_CHIRP)
    print(f"{start_lag = }, {end_lag = }, {difference = }, {expected_difference = }")

    if plot_correlations:
        fig, ax = plt.subplots()
        prefix, suffix = 300, 500
        time_vals = np.arange(-prefix, suffix) / FS
        ax.plot(
            time_vals, start_correlation[start_lag - prefix : start_lag + suffix], label="Correlation with start chirp"
        )
        ax.plot(time_vals, end_correlation[end_lag - prefix : end_lag + suffix], label="Correlation with end chirp")
        ax.legend()
        ax.set_xlabel("Time (seconds)")

    aligned_recv_signal = np.roll(recv_signal, -start_lag)
    return aligned_recv_signal[: expected_signal_len + 2 * END_CHIRP.size]
