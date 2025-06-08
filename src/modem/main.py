import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.signal
from modem import chirp, pilot, qpsk, wav, estimate, freq

from modem.constants import (
    BYTES_BLOCK_LENGTH,
    DATA_PER_PILOT,
    DATA_BLOCK_LENGTH,
    LOWER_FREQUENCY_BOUND,
    OFDM_SYMBOL_LENGTH,
    RNG_SEED,
    UPPER_FREQUENCY_BOUND,
    ldpc_standard,
    ldpc_rate,
    ldpc_z_val,
    ldpc_ptype,
    ldpc_dec_type,
    corr_factor,
    FS,
    LDPC_INPUT_LENGTH,
    LDPC_OUTPUT_LENGTH,
)  ###
from modem.chirp import synchronise  ###-
from ldpc_jossy.py.ldpc import code  ###


### = Ido's addition
ldpc_code = code(standard=ldpc_standard, rate=ldpc_rate, z=ldpc_z_val, ptype=ldpc_ptype)
proto = ldpc_code.assign_proto()
pcmat = ldpc_code.pcmat()


def pad_to_multiple(bit_array: npt.NDArray[np.uint8], block_length: int) -> npt.NDArray[np.uint8]:
    """Pad bit array of 0/1 to multiple of block length"""
    (length,) = bit_array.shape
    remainder = length % block_length
    pad_len = block_length - remainder if remainder != 0 else 0
    rng = np.random.default_rng(seed=RNG_SEED)
    padding = rng.integers(0, 1, size=pad_len, dtype=np.uint8)

    return np.concatenate((bit_array, padding))


def encode_data(data: bytes) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))  # convert bytes to bits
    # Pad to an even number of OFDM symbols (two LDPC blocks per symbol)
    data_bits = pad_to_multiple(data_bits, DATA_BLOCK_LENGTH).reshape(-1, LDPC_INPUT_LENGTH)
    coded_data_bits = np.vstack(
        [ldpc_code.encode(ldpc_block) for ldpc_block in data_bits]
    )  # PERFORMS LDPC encoding on each block of input bits and stacks the results. Applies it on each row and then stacks the rows.
    data_qpsk_values = qpsk.qpsk_encode(
        coded_data_bits.flatten()
    )  # Encodes the coded data bits into qpsk symbols based on the gray code.
    data_ofdm_symbols = qpsk.encode_data_ofdm_symbol(
        data_qpsk_values
    )  # Encode the ofdm symbols from the qpsk symbols (i.e tke the IFFT)

    num_ofdm_blocks = data_ofdm_symbols.shape[0]  # Number of OFDM blocks is the number of rows in data_ofdm_symbols
    print(f"Data encoded into {num_ofdm_blocks} OFDM blocks")

    pilot_qpsk_symbols = pilot.generate_pilot_blocks(
        math.ceil(num_ofdm_blocks / DATA_PER_PILOT)
    )  # generate the pilto blocks
    pilot_ofdm_symbols = qpsk.encode_data_ofdm_symbol(
        pilot_qpsk_symbols.flatten()
    )  # Encode the pilot blocks into OFDM symbols
    ofdm_symbols = pilot.interleave_pilot_blocks(
        data_ofdm_symbols, pilot_ofdm_symbols
    )  # Interleave the pilot blocks with the data blocks, so that each pilot block is followed by DATA_PER_PILOT data blocks.
    print(f"Transimission is {ofdm_symbols.shape[0]} OFDM symbols total")

    signal = np.concatenate(
        (chirp.START_CHIRP, ofdm_symbols.flatten(), chirp.END_CHIRP)
    )  # Concatenate the start chirp, the flattened OFDM symbols and the end chirp to create the final signal.
    return signal, data_qpsk_values.reshape(-1, DATA_BLOCK_LENGTH)


def decode_data(
    signal: npt.NDArray[np.float64], sent_data_qpsk_values: npt.NDArray[np.complex128] | None = None, plot: bool = False
) -> bytes:
    aligned_signal = chirp.synchronise(
        signal, plot_correlations=plot, plot_spectrogram=plot
    )  # Synchronise the received signal with the start chirp.
    recv_ofdm_symbols = np.reshape(
        aligned_signal[chirp.START_CHIRP.size : -chirp.END_CHIRP.size], (-1, OFDM_SYMBOL_LENGTH)
    )  # Reshape into matrix of OFDM symbols, excluding the start and end chirps.

    received_QPSK = qpsk.decode_ofdm_symbol(recv_ofdm_symbols)
    data_blocks, pilot_blocks = pilot.extract_pilot_blocks(received_QPSK)

    sent_pilot_qpsk_symbols = pilot.generate_pilot_blocks(pilot_blocks.shape[0])
    known_qpsk_symbols = pilot.interleave_pilot_blocks(
        np.full(data_blocks.shape, np.nan, dtype=np.complex128), sent_pilot_qpsk_symbols
    )

    with np.errstate(invalid="ignore"):
        observed_frequency_gains = qpsk.decode_ofdm_symbol(recv_ofdm_symbols) / known_qpsk_symbols
    freq_gains = freq.get_freq_gains(observed_frequency_gains, plot=plot)
    data_freq_gains, pilot_freq_gains = pilot.extract_pilot_blocks(freq_gains)

    snr, noise_var = estimate.estimate_snr(known_qpsk_symbols, qpsk.decode_ofdm_symbol(recv_ofdm_symbols), freq_gains)
    print(f"SNR ratio is {10 * np.log10(snr):.5f}dB")

    adjusted_data_qpsk_symbols = qpsk.wiener_filter(data_blocks, data_freq_gains, snr)

    if plot:
        qpsk.plot_received_constellation(adjusted_data_qpsk_symbols, sent_data_qpsk_values)

    llr = estimate.find_LLRs(adjusted_data_qpsk_symbols, data_freq_gains, noise_var)

    decode_output = [ldpc_code.decode(chunk.copy(), ldpc_dec_type, corr_factor) for chunk in llr]
    decoded_llr, iterations = map(np.array, zip(*decode_output))
    print(f"LDPC decoding finished, all blocks took between {np.min(iterations)} and {np.max(iterations)} iterations")
    bits = (decoded_llr < 0).astype(int)
    print(
        f"Fraction of LLRs updated {np.mean(llr != decoded_llr)}, fraction of LLR bits flipped {np.mean((llr > 0) != (decoded_llr > 0))}"
    )

    if sent_data_qpsk_values is not None:
        sent_bits = estimate.find_LLRs(sent_data_qpsk_values, np.array(1, dtype=np.complex128), 1) < 0
        initial_llr_bit_error_rate = np.mean(sent_bits != (llr < 0), axis=1)
        final_llr_bit_error_rate = np.mean(sent_bits != bits, axis=1)
        print(f"Pre-LDPC LLR bit error rate {initial_llr_bit_error_rate}")
        print(f"Post-LDPC LLR bit error rate {final_llr_bit_error_rate}")

    # Discard parity bits from systematic LDPC coding
    data_bits = bits[:, :LDPC_INPUT_LENGTH]

    return np.packbits(data_bits).tobytes()


def encode_file(filepath: str, contents: bytes) -> bytes:
    size = len(contents)
    data = filepath.encode("ascii") + b"\0" + str(size).encode("ascii") + b"\0" + contents
    return data


def decode_file(data: bytes) -> tuple[str, bytes]:
    filename_end = data.find(b"\0")
    filename = data[:filename_end].decode(encoding="ascii", errors="replace")
    data = data[filename_end + 1 :]

    file_length_end = data.find(b"\0")
    file_length = int(data[:file_length_end].decode(encoding="ascii", errors="replace"))
    data = data[file_length_end + 1 :]
    return filename, data[:file_length]


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    data = bytes(rng.integers(256, size=BYTES_BLOCK_LENGTH * 50, dtype=np.uint8))
    signal, data_qpsk_values = encode_data(data)
    wav.generate_wav("signal.wav", signal)

    recv_signal = wav.read_wav("2025-06-05_LT5_2.wav")
    decode_data(recv_signal, data_qpsk_values, False)
    plt.show()
