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


def encode_data(data: bytes) -> npt.NDArray[np.float64]:
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)) # convert bytes to bits
    pad_len = int((-len(data_bits)) % LDPC_INPUT_LENGTH) # calculates padding length required to make total length of data_bits an exact multiple of LDDPC_INPUT_LENGTH
    data_bits = np.pad(data_bits, (0, pad_len), constant_values=0).reshape((-1, LDPC_INPUT_LENGTH)) # Pads data_bits with zeros at its end ensuring total length is a mutliple of LDPC_input_Length. THen ti is shaped into a a 2d array where each row has LDPC INPUT_LENGTH BITS.
    coded_data_bits = np.vstack([ldpc_code.encode(ldpc_block) for ldpc_block in data_bits])# PERFORMS LDPC encoding on each block of input bits and stacks the results. Applies it on each row and then stacks the rows.
    data_qpsk_values = qpsk.qpsk_encode(coded_data_bits.flatten()) # Encodes the coded data bits into qpsk symbols based on the gray code.
    data_ofdm_symbols = qpsk.encode_ofdm_symbol(data_qpsk_values) # Encode the ofdm symbols from the qpsk symbols (i.e tke the IFFT)

    num_ofdm_blocks = data_ofdm_symbols.shape[0] # Number of OFDM blocks is the number of rows in data_ofdm_symbols
    print(f"Data encoded into {num_ofdm_blocks} OFDM blocks")

    pilot_qpsk_symbols = pilot.generate_pilot_blocks(int(math.ceil(num_ofdm_blocks / DATA_PER_PILOT))) # generate the pilto blocks
    pilot_ofdm_symbols = qpsk.encode_ofdm_symbol(pilot_qpsk_symbols.flatten()) # Encode the pilot blocks into OFDM symbols
    ofdm_symbols = pilot.interleave_pilot_blocks(data_ofdm_symbols, pilot_ofdm_symbols) # Interleave the pilot blocks with the data blocks, so that each pilot block is followed by DATA_PER_PILOT data blocks.

    signal = np.concatenate((chirp.START_CHIRP, ofdm_symbols.flatten(), chirp.END_CHIRP)) # Concatenate the start chirp, the flattened OFDM symbols and the end chirp to create the final signal.
    return signal


def decode_data(signal: npt.NDArray[np.float64], sent_data_qpsk_values: npt.NDArray[np.complex128] | None = None,
                plot: bool = False) -> None:
    aligned_signal = chirp.synchronise(signal, plot_correlations=plot, plot_spectrogram=plot) # Synchronise the received signal with the start chirp.
    recv_ofdm_symbols = np.reshape(
        aligned_signal[chirp.START_CHIRP.size : -chirp.END_CHIRP.size], (-1, OFDM_SYMBOL_LENGTH)
    ) # Reshape into matrix of OFDM symbols, excluding the start and end chirps.
    
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

    sent_data_qpsk_values = sent_data_qpsk_values.reshape(-1, DATA_BLOCK_LENGTH)
    
    if plot:
        qpsk.plot_received_constellation(adjusted_data_qpsk_symbols, sent_data_qpsk_values)

    bits_real = (np.real(sent_data_qpsk_values.flatten()) < 0).astype(int)
    bits_imag = (np.imag(sent_data_qpsk_values.flatten()) < 0).astype(int)
    bits = np.column_stack((bits_imag, bits_real)).reshape(-1)
    bits = bits[:(bits.size // LDPC_OUTPUT_LENGTH) * LDPC_OUTPUT_LENGTH]
    sent_bits = bits.reshape(-1, LDPC_OUTPUT_LENGTH)

    llr = estimate.find_LLRs(adjusted_data_qpsk_symbols, data_freq_gains, noise_var)

    decode_output = [ldpc_code.decode(chunk.copy(), ldpc_dec_type, corr_factor) for chunk in llr]
    decoded_llr, iterations = map(np.array, zip(*decode_output))
    print(f"LDPC decoding finished, all blocks took between {np.min(iterations)} and {np.max(iterations)} iterations")
    bits = (decoded_llr < 0).astype(int)

    change_in_llrs = decoded_llr - llr
    max_index = np.argmax(np.abs(change_in_llrs))
    min_index = np.argmin(np.abs(change_in_llrs))
    print(f"Change in LLRs: max {np.max(np.abs(decoded_llr - llr))} from {llr.flatten()[max_index]} to {decoded_llr.flatten()[max_index]}")
    print(f"Change in LLRs: min {np.min(np.abs(decoded_llr - llr))} from {llr.flatten()[min_index]} to {decoded_llr.flatten()[min_index]}")
    print(f"Fraction of LLRs updated {np.mean(llr != decoded_llr)}, fraction of LLR bits flipped {np.mean((llr > 0) != (decoded_llr > 0))}")
    # Expected to be identical to quadrant-based results
    initial_llr_bit_error_rate = np.mean(sent_bits != (llr < 0).astype(int), axis=1)
    final_llr_bit_error_rate = np.mean(sent_bits != bits, axis=1)
    print(f"Pre-LDPC LLR bit error rate {initial_llr_bit_error_rate}")
    print(f"Post-LDPC LLR bit error rate {final_llr_bit_error_rate}")

    return bits



if __name__ == "__main__":
    #rng = np.random.default_rng(seed=42)
    #data = bytes(rng.integers(256, size=BYTES_BLOCK_LENGTH * 200, dtype=np.uint8))
    #signal = encode_data(data)
    #wav.generate_wav("signal.wav", signal)
    def generate_test_data():
    # Generate random binary data
        rng = np.random.default_rng(seed=RNG_SEED)  # Use a fixed seed for reproducibility
        data = bytes(rng.integers(0, 256, size=BYTES_BLOCK_LENGTH*50, dtype=np.uint8))  # Example: 10 blocks of data

        # Encode the data using the encode_data function
        data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        pad_len = int((-len(data_bits)) % LDPC_INPUT_LENGTH)
        data_bits = np.pad(data_bits, (0, pad_len), constant_values=0).reshape((-1, LDPC_INPUT_LENGTH))
        coded_data_bits = np.vstack([ldpc_code.encode(ldpc_block) for ldpc_block in data_bits])
        data_qpsk_values = qpsk.qpsk_encode(coded_data_bits.flatten())

        return data, data_qpsk_values, coded_data_bits

# Generate and save the test dataset
    data, data_qpsk_values, coded_data_bits = generate_test_data()
    # signal = encode_data(data)
    # wav.generate_wav("signal.wav", signal)
    # print("1...")
    
    recv_signal = wav.read_wav("2025-06-05_LT5_2.wav")
    decode_data(recv_signal, data_qpsk_values, True)
    plt.show()

