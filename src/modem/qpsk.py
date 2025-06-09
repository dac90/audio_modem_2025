import numpy as np
import numpy.typing as npt
from ldpc_jossy.py.ldpc import code
import matplotlib.pyplot as plt
import scipy.signal
from modem import qpsk, chirp, wav
import sounddevice as sd
import scipy.io.wavfile as wav
DURATION = 15  # Duration of the recording in seconds
FS = 48000 
import os

from modem.constants import (
    CYCLIC_PREFIX_LENGTH,
    FFT_BLOCK_LENGTH,
    QPSK_BLOCK_LENGTH,
    QPSK_MULTIPLIER,
    POSITIVE_LOWER_BIN,
    POSITIVE_UPPER_BIN,
    DATA_BLOCK_LENGTH,
    ldpc_standard,
    ldpc_rate,
    ldpc_z_val,
    ldpc_ptype,
)


ldpc_code = code(standard=ldpc_standard, rate=ldpc_rate, z=ldpc_z_val, ptype=ldpc_ptype)
proto = ldpc_code.assign_proto()
pcmat = ldpc_code.pcmat()
num_variable_nodes = ldpc_code.Nv
print(num_variable_nodes)


def qpsk_encode(bits: npt.NDArray[np.uint8]) -> npt.NDArray[np.complex128]:
    """Encode bytes into constellation symbols
    using QPSK in the frequency domain, before OFDM.

    Parameters
    ----------
    bits : npt.NDArray[np.bool]
        Array with bits to be encoded.

    Returns
    ----------
    qpsk_values : npt.NDArray[np.complex128]
        Returns 1D array of QPSK constellation symbols of size (n * DATA_BLOCK_LENGTH),
        where n = (len(bits) // (2 * DATA_BLOCK_LENGTH))
    """
    bit_pairs = bits.reshape(-1, 2) # groups the bits into pairs of two.
    gray_indices = (bit_pairs[:, 0] << 1) | bit_pairs[:, 1] 

    qpsk_constellation = np.array(
        [
            (1 + 1j),  # 00
            (1 - 1j),  # 01
            (-1 + 1j),  # 10
            (-1 - 1j),  # 11
        ]
    )

    qpsk_values = (QPSK_MULTIPLIER / np.sqrt(2)) * qpsk_constellation[gray_indices]
    return qpsk_values


def qpsk_decode(values: npt.NDArray[np.complex128],) -> bytes:
    """Decode received constellation symbols
    in the frequency domain into bytes using QPSK"""
    values = np.asarray(values)
    bits_real = (np.real(values) < 0).astype(int) # converts the real part of values to bits, if it is negative then it is 1, if it is positive then it is zero.
    bits_imag = (np.imag(values) < 0).astype(int) # converts the imaginary part of values to bits, if it is negative then it is 1, if it is positive then it is zero.
    bits = np.column_stack((bits_imag, bits_real)).reshape(-1) # stacks the bits together in pairs and reshapes them into a 1D array.
    bytes = np.packbits(bits).tobytes() # converts the bits to bytes.
    return bytes
import numpy as np

# Define the QPSK constellation symbols
qpsk_symbols = np.array([1+1j, 1-1j, -1-1j, -1+1j], dtype=np.complex128)

# Generate a random array of 12,000 symbols
random_symbols = np.random.choice(qpsk_symbols, size=12000)

def encode_ofdm_symbol(
    qpsk_values: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float64]:
    """Encode constellation symbols in frequency-domain
    into time-domain OFDM symbol for transmission"""

    #Input: qpsk_values is a 1D array of QPSK symbols (1+j,1-j,-1+j,-1-j...1+j,-1-j,1+j,-1-j) 
    # Output : Multiple arrays of QPSK symbols, each of with qpsk data of size QPSK_BLOCK_LENGTH_DATA and 
    #          zeroes of length POSITIVE_LOWER_BIN and POSITIVE_UPPER_BIN at the beginning and end of the array.
    assert qpsk_values.shape[0] % QPSK_BLOCK_LENGTH == 0, f"QPSK values need to be an even multiple of OFDM length, got {qpsk_values.shape[0]}"
    qpsk_blocks = qpsk_values.reshape(-1, QPSK_BLOCK_LENGTH) # matrix with "DATA_block_LENGTH" Columns"

    conj_blocks = np.conj(np.fliplr(qpsk_blocks)) # conjugate and flip them 
    zero_col = np.zeros((np.shape(qpsk_blocks)[0], 1), dtype=np.complex128)
    X = np.hstack([zero_col, qpsk_blocks, zero_col, conj_blocks]) # create the full matrix with zeroes at bin 0 and N/2

    x = np.fft.ifft(X, n=FFT_BLOCK_LENGTH) # take the IFFT to get time domain OFDM symbol
    x = np.hstack([x[:, -CYCLIC_PREFIX_LENGTH:], x]) # Add cyclic prefix to the OFDM symbol
    np.testing.assert_allclose(np.imag(x), np.zeros_like(x), atol=1e-14)

    return np.real(x) # return the real part in case we have small imaginary values due to numerical errors


def encode_data_ofdm_symbol(
    qpsk_values: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float64]:
    """Encode constellation symbols in frequency-domain
    into time-domain OFDM symbol for transmission"""

    #Input: qpsk_values is a 1D array of QPSK symbols (1+j,1-j,-1+j,-1-j...1+j,-1-j,1+j,-1-j) 
    # Output : Multiple arrays of QPSK symbols, each of with qpsk data of size QPSK_BLOCK_LENGTH_DATA and 
    #          zeroes of length POSITIVE_LOWER_BIN and POSITIVE_UPPER_BIN at the beginning and end of the array.
    assert qpsk_values.shape[0] % DATA_BLOCK_LENGTH == 0, f"QPSK values need to be an even multiple of OFDM length, got {qpsk_values.shape[0]}"
    qpsk_data_blocks = qpsk_values.reshape(-1, DATA_BLOCK_LENGTH) # matrix with "DATA_block_LENGTH" Columns"
    qpsk_blocks = np.zeros((qpsk_data_blocks.shape[0], FFT_BLOCK_LENGTH//2), dtype=np.complex128) # FIll the matrix with zeroes.
    qpsk_blocks[:, POSITIVE_LOWER_BIN:POSITIVE_UPPER_BIN] = qpsk_data_blocks # Fill the matrix with the QPSK data blocks int he middle of the matrix.
    qpsk_blocks = qpsk_blocks[:, 1:] # Remove the first column of zeroes, as it is not needed for the IFFT.

    return encode_ofdm_symbol(qpsk_blocks.flatten())


def decode_ofdm_symbol(
    ofdm_symbol: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex128]:
    """Decode time-domain OFDM symbol into
    constellation symbols in frequency-domain"""
    ofdm_symbol = ofdm_symbol[:, CYCLIC_PREFIX_LENGTH:]  # Discard cyclic prefix
    freq_values = np.fft.fft(ofdm_symbol) # take the FFT to get frequency domain values
    freq_values = freq_values[:, POSITIVE_LOWER_BIN:POSITIVE_UPPER_BIN] # select the positive frequencies. These are our data points.

    return freq_values

def wiener_filter(y: npt.NDArray[np.complex128], h: npt.NDArray[np.complex128], snr: float):
    x = (y * np.conj(h)) / (np.abs(h) ** 2 + (1 / snr))
    return x


def plot_received_constellation(recv_data_qpsk_values: npt.NDArray[np.complex128],
                                sent_data_qpsk_values: npt.NDArray[np.complex128] | None = None):
    """Create Argand plot of normalised received QPSK symbols for first 5 and last 5 blocks.
    Colors symbols depending on original sent symbol if sent QPSK symbols are provided."""
    fig, axs = plt.subplots(2, 5)
    block_indices = [*range(5), *range(-5, 0)]  # Plot first 5 and last 5 blocks
    ax_limit = np.max(np.maximum(np.abs(recv_data_qpsk_values.real), np.abs(recv_data_qpsk_values.imag)))
    for block_index, ax in zip(block_indices, axs.flatten(), strict=True):
        recv_block = recv_data_qpsk_values[block_index]

        if sent_data_qpsk_values is not None:
            sent_block = sent_data_qpsk_values[block_index]
            positive_real_mask = np.real(sent_block) > 0
            positive_imag_mask = np.imag(sent_block) > 0
            mask_00 = positive_real_mask & positive_imag_mask
            mask_01 = (~positive_real_mask) & positive_imag_mask
            mask_11 = (~positive_real_mask) & (~positive_imag_mask)
            mask_10 = positive_real_mask & (~positive_imag_mask)

            for mask, bits in ((mask_00, "00"), (mask_01, "01"), (mask_10, "10"), (mask_11, "11")):
                ax.scatter(np.real(recv_block[mask]), np.imag(recv_block[mask]), label=bits)
            ax.legend()
        else:
            ax.scatter(np.real(recv_block), np.imag(recv_block))

        ax.set_xlim(-ax_limit, ax_limit)
        ax.set_ylim(-ax_limit, ax_limit)

        # Convert negative index to positive
        block_num = block_index if block_index >= 0 else block_index + recv_data_qpsk_values.shape[0]
        ax.set_title(f"OFDM block {block_num + 1}")


########################################
"""Implement transmitter and reciever
"""

def record_audio(duration, FS):
    print("Recording...")
    audio = sd.rec(int((duration) * FS), samplerate=FS, channels=2, dtype='int16') # channels = 2 might need changing
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
    return audio

# Function to save audio to a WAV file
def save_audio(filename, audio, FS):
    wav.write(filename, FS, audio)
    print(f"Audio saved to {filename}")

def recieve_signal():
    output_file = "testgroup4.wav"  # Name of the output WAV file
    recorded_audio = record_audio(DURATION, FS)
    save_audio(output_file, recorded_audio, FS)
    _,  recieved_signal = scipy.io.wavfile.read("testgroup4.wav")

    return recieved_signal