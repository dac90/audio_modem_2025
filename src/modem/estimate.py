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
    noise_power = np.nansum(np.abs(error) ** 2, axis=0)
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


def estimate_noise_var(received_symbols: npt.NDArray[np.complex128],
                       channel_gains: npt.NDArray[np.complex128],
                       expected_symbols: npt.NDArray[np.complex128],):
    # estimate the noise variance by computing the complex
    #noise N = Y −ckX for each observation and averaging its squared real and imaginary
    #parts.
    noise = received_symbols - (channel_gains * expected_symbols)
    noise_var = np.mean(np.square(np.real(noise))) + np.mean(np.square(np.imag(noise)))
    return noise_var
        
def find_LLRs(
    received_symbols: npt.NDArray[np.complex128],
    channel_gains: npt.NDArray[np.complex128],
    noise_var: float,
) -> npt.NDArray[np.float64]:
    """Calculate the Log-Likelihood Ratios (LLRs) for each symbol."""
    # Calculate Equalised Y symbols, find real and imaginary, find conjugate of channel.
    equalised_received = received_symbols / channel_gains 
    real_equalised_received = np.real(equalised_received)
    imag_equalised_received = np.imag(equalised_received)
    channel_gains_conjugate = np.conjugate(channel_gains)
    # Calculate the LLRs
    llrs_real = (np.sqrt(2)*channel_gains_conjugate*channel_gains*real_equalised_received / (noise_var)).astype(np.float64)

    llrs_imag = (np.sqrt(2)*channel_gains_conjugate*channel_gains*imag_equalised_received / (noise_var)).astype(np.float64)

    return llrs_imag, llrs_real


##############################################
"""
received_symbols = np.array([1+1j, 2+2j, 3+3j])
channel_gains = np.array([1+0j, 0.5+0.5j, 1-1j])

noise_var = 0.1
li,lr = find_LLRs(received_symbols, channel_gains, noise_var)
print("LLRs Real:", lr)
print("LLRs Imaginary:", li)
"""