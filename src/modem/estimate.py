import numpy as np
import numpy.typing as npt


def estimate_snr(
    expected_symbols: npt.NDArray[np.complex128],
    received_symbols: npt.NDArray[np.complex128],
    channel_gains: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    """Calculate the SNR for each frequency (ratio not dB).
    Note this function expects the received symbols before
    gain compensation."""
    # TODO: probably best to just use an input with unit power
    signal_power = np.abs(expected_symbols * channel_gains) ** 2
    error = (expected_symbols * channel_gains) - received_symbols
    noise_power = np.sum(np.abs(error) ** 2)
    return signal_power / noise_power


def avg_phase_shift(
    expected_symbols: npt.NDArray[np.complex128], received_symbols: npt.NDArray[np.complex128]
) -> float:
    """Return average phase shift of symbols in radians"""
    expected_phase = np.angle(expected_symbols)
    received_phase = np.angle(received_symbols)
    return np.mean(received_phase - expected_phase)


def estimate_channel(
    ofdm_estimates: list[npt.NDArray[np.complex128] | None],
    start_chirp_estimation: npt.NDArray[np.complex128],
    end_chirp_estimation: npt.NDArray[np.complex128],
    desired_estimate_index: int,
) -> npt.NDArray[np.complex128]: ...
