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

    pilot_qpsk_symbols = pilot.generate_pilot_blocks(num_ofdm_blocks + 1) # generate the pilto blocks
    pilot_ofdm_symbols = qpsk.encode_ofdm_symbol(pilot_qpsk_symbols) # Encode the pilot blocks into OFDM symbols
    ofdm_symbols = pilot.interleave_pilot_blocks(data_ofdm_symbols, pilot_ofdm_symbols) # Interleave the pilot blocks with the data blocks, so that each pilot block is followed by DATA_PER_PILOT data blocks.

    signal = np.concatenate((chirp.START_CHIRP, ofdm_symbols.flatten(), chirp.END_CHIRP)) # Concatenate the start chirp, the flattened OFDM symbols and the end chirp to create the final signal.
    return signal


def decode_data(pilot_qpsk_symbols: npt.NDArray[np.complex128],data_qpsk_values: npt.NDArray[np.complex128],signal: npt.NDArray[np.float64], plot: bool = False) -> None:
    aligned_signal = chirp.synchronise(signal, plot_correlations=plot) # Synchronise the received signal with the start chirp.
    recv_ofdm_symbols = np.reshape(
        aligned_signal[chirp.START_CHIRP.size : -chirp.END_CHIRP.size], (-1, OFDM_SYMBOL_LENGTH)
    ) # Reshape into matrix of OFDM symbols, exlcuding the start and end chirps.
    

    if plot:
        fig, ax = plt.subplots()

        f, t_spec, Sxx = scipy.signal.spectrogram(aligned_signal, FS)

        pcm = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading="gouraud")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [sec]")
        cbar = fig.colorbar(pcm, ax=ax, label="Power/Frequency (dB/Hz)")

    num_ofdm_symbols = recv_ofdm_symbols.shape[0] 
    length_ofdm_symbol = recv_ofdm_symbols.shape[1] 
    

    """assert num_ofdm_symbols % 2 == 1, f"Expected an odd number of total symbols"
    num_data_symbols = num_ofdm_symbols // 4 
    num_pilot_symbols = num_data_symbols + 1 # Fix this 
    pilot_qpsk_symbols = np.reshape(pilot.generate_pilot_blocks(num_pilot_symbols), (-1, DATA_BLOCK_LENGTH))"""
    known_qpsk_symbols = pilot.interleave_pilot_blocks(
        np.full((num_data_symbols, DATA_BLOCK_LENGTH), np.nan, dtype=np.complex128), pilot_qpsk_symbols
    )
    ###
    recieved_QPSK = qpsk.decode_ofdm_symbol(recv_ofdm_symbols)
    data_blocks,pilot_blocks = pilot.extract_pilot_blocks(recieved_QPSK)
    ###
    with np.errstate(invalid="ignore"):
        observed_frequency_gains = qpsk.decode_ofdm_symbol(recv_ofdm_symbols) / known_qpsk_symbols
    freq_gains = freq.get_freq_gains(observed_frequency_gains, plot=plot)
    data_freq_gains, pilot_freq_gains = pilot.extract_pilot_blocks(freq_gains)

    snr_estimates, noise_var = estimate.estimate_snr(known_qpsk_symbols, qpsk.decode_ofdm_symbol(recv_ofdm_symbols), freq_gains)
    avg_snr = np.nanmean(snr_estimates, axis=0)

    print(f"SNR ratio is {10 * np.log10(np.nanmean(avg_snr)):.5f}dB")

    if plot:
        fig, ax = plt.subplots()
        freqs = np.linspace(LOWER_FREQUENCY_BOUND, UPPER_FREQUENCY_BOUND, DATA_BLOCK_LENGTH, endpoint=False)
        for i, block_gain in enumerate(observed_frequency_gains[~np.isnan(observed_frequency_gains).any(axis=1)]):
            ax.plot(freqs, np.log10(np.abs(block_gain)), label=f"Block {i}")
        ax.plot(freqs, np.mean(np.log10(np.abs(freq_gains)), axis=0), label="Mean")
        ax.set_title("Frequency Gain Plot (in dB)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.legend()

    recv_data_qpsk_symbols = qpsk.decode_ofdm_symbol(recv_ofdm_symbols[1::2, :])
    adjusted_data_qpsk_symbols = qpsk.wiener_filter(recv_data_qpsk_symbols, data_freq_gains, avg_snr)
    # Without Wiener filter, just using ratio
    # adjusted_data_qpsk_symbols = qpsk.decode_ofdm_symbol(recv_ofdm_symbols[1::2, :]) / data_freq_gains

    data_qpsk_values = data_qpsk_values.reshape(-1, DATA_BLOCK_LENGTH)
    
    if plot:
        fig, axs = plt.subplots(2, 5)
        for sent_vals, recv_vals, ax in zip(data_qpsk_values[:10], adjusted_data_qpsk_symbols[:10], axs.flatten(), strict=True):
            positive_real_mask = np.real(sent_vals) > 0
            positive_imag_mask = np.imag(sent_vals) > 0
            mask_00 = positive_real_mask & positive_imag_mask
            mask_01 = (~positive_real_mask) & positive_imag_mask
            mask_11 = (~positive_real_mask) & (~positive_imag_mask)
            mask_10 = positive_real_mask & (~positive_imag_mask)

            for mask, bits in ((mask_00, "00"), (mask_01, "01"), (mask_10, "10"), (mask_11, "11")):
                ax.scatter(np.real(recv_vals[mask]), np.imag(recv_vals[mask]), label=bits)

            ax.legend()

    # Alternative LLR calculation
    # llr_real = np.real(np.sqrt(2) * adjusted_data_qpsk_symbols * avg_snr)
    # llr_imag = np.imag(np.sqrt(2) * adjusted_data_qpsk_symbols * avg_snr)
    bits_real = (np.real(data_qpsk_values.flatten()) < 0).astype(int)
    bits_imag = (np.imag(data_qpsk_values.flatten()) < 0).astype(int)
    bits = np.column_stack((bits_imag, bits_real)).reshape(-1)
    bits = bits[:(bits.size // LDPC_OUTPUT_LENGTH) * LDPC_OUTPUT_LENGTH]
    sent_bits = bits.reshape(-1, LDPC_OUTPUT_LENGTH)

    #_,noise_var = estimate.estimate_noise_var(recv_data_qpsk_symbols, data_freq_gains, data_qpsk_values)
    llr_imag, llr_real = estimate.find_LLRs(adjusted_data_qpsk_symbols, data_freq_gains, noise_var)

    llr = np.column_stack((llr_imag.flatten(), llr_real.flatten())).reshape(-1)
    llr = llr[:(llr.size // LDPC_OUTPUT_LENGTH) * LDPC_OUTPUT_LENGTH].reshape(-1, LDPC_OUTPUT_LENGTH)

    decode_output = [ldpc_code.decode(chunk.copy(), ldpc_dec_type, corr_factor) for chunk in llr]
    decoded_llr, iterations = zip(*decode_output)
    print(f"{iterations = }")
    decoded_llr = np.array(decoded_llr)
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
        data = bytes(rng.integers(0, 256, size=BYTES_BLOCK_LENGTH*5, dtype=np.uint8))  # Example: 10 blocks of data

        # Encode the data using the encode_data function
        data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        pad_len = int((-len(data_bits)) % LDPC_INPUT_LENGTH)
        data_bits = np.pad(data_bits, (0, pad_len), constant_values=0).reshape((-1, LDPC_INPUT_LENGTH))
        coded_data_bits = np.vstack([ldpc_code.encode(ldpc_block) for ldpc_block in data_bits])
        data_qpsk_values = qpsk.qpsk_encode(coded_data_bits.flatten())

        return data, data_qpsk_values, coded_data_bits

# Generate and save the test dataset
    data, data_qpsk_values, coded_data_bits = generate_test_data()
    print("1...")
    
    recv_signal = wav.read_wav(r"C:\Users\idoba\Downloads\2025-06-03_LT5.wav")
    decode_data(data_qpsk_values, recv_signal, True )
    print(f"shape of encoded_data_bits: {coded_data_bits.shape}")
    plt.show()


