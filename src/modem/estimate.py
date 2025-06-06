import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from modem.constants import LDPC_OUTPUT_LENGTH, QPSK_MULTIPLIER


def estimate_snr(
    expected_symbols: npt.NDArray[np.complex128],
    received_symbols: npt.NDArray[np.complex128],
    channel_gains: npt.NDArray[np.complex128],
    plot: bool = False,
) -> tuple[float, float]:
    """Calculate the SNR, A^2/sigma^2.
    Note this function expects the received symbols before
    gain compensation."""
    error = (expected_symbols * channel_gains) - received_symbols

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(np.real(error), np.imag(error))
        ax.set_title("Error $hX - Y$")
        ax.set_aspect('equal')

    signal_power = QPSK_MULTIPLIER ** 2
    noise_power = float(np.nanmean(np.abs(error) ** 2))
    snr = signal_power / noise_power
    var = noise_power
    return snr, var

        
def find_LLRs(
    received_symbols: npt.NDArray[np.complex128],
    channel_gains: npt.NDArray[np.complex128],
    noise_var: float,
) -> npt.NDArray[np.float64]:
    """Calculate the Log-Likelihood Ratios (LLRs) for each symbol,
    based on normalised received symbols, channel gains, and variance of complex additive noise
    (single value since not normalised by frequency).

    Expect received_symbols to be 2D array of normalised received QPSK symbols.
    Normally distributed with mean of QPSK_MULTIPLIER and variance (noise_var / 2*h^2)
    where noise_var is the variance on the complex Gaussian noise.
    Equivalent to y' in Jossy's paper, although note factor of 2 difference in how variance is defined.
    """
    complex_llrs = np.sqrt(2) * np.abs(channel_gains) ** 2 * QPSK_MULTIPLIER * received_symbols / (noise_var / 2)

    # From N x DATA_BLOCK_LENGTH array of complex LLRs to 2N x DATA_BLOCK_LENGTH array of imaginary and real LLRs
    llrs = np.column_stack((complex_llrs.flatten().imag, complex_llrs.flatten().real)).reshape(-1, complex_llrs.shape[1])
    
    # Reshape into 2N X LDPC_OUTPUT_LENGTH array for LDPC decoding
    llrs = llrs.flatten().reshape(-1, LDPC_OUTPUT_LENGTH)

    return llrs
