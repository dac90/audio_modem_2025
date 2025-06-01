import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.signal
from modem import chirp, pilot, qpsk, wav, estimate

from modem.constants import (
    BYTES_BLOCK_LENGTH,
    DATA_PER_PILOT,
    DATA_BLOCK_LENGTH,
    LOWER_FREQUENCY_BOUND,
    OFDM_SYMBOL_LENGTH,
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
from modem.ldpc import code  ###



### = Ido's addition
ldpc_code = code(standard=ldpc_standard, rate=ldpc_rate, z=ldpc_z_val, ptype=ldpc_ptype)
proto = ldpc_code.assign_proto()
pcmat = ldpc_code.pcmat()


def encode_data(data: bytes) -> npt.NDArray[np.float64]:
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    pad_len = int((-len(data_bits)) % LDPC_INPUT_LENGTH)
    data_bits = np.pad(data_bits, (0, pad_len), constant_values=0).reshape((-1, LDPC_INPUT_LENGTH))
    coded_data_bits = np.vstack([ldpc_code.encode(ldpc_block) for ldpc_block in data_bits])
    data_qpsk_values = qpsk.qpsk_encode(coded_data_bits.flatten())
    data_ofdm_symbols = qpsk.encode_ofdm_symbol(data_qpsk_values)

    num_ofdm_blocks = data_ofdm_symbols.shape[0]
    print(f"Data encoded into {num_ofdm_blocks} OFDM blocks")

    pilot_qpsk_symbols = pilot.generate_pilot_blocks(num_ofdm_blocks + 1)
    pilot_ofdm_symbols = qpsk.encode_ofdm_symbol(pilot_qpsk_symbols)
    ofdm_symbols = pilot.interleave_pilot_blocks(data_ofdm_symbols, pilot_ofdm_symbols)

    signal = np.concatenate((chirp.START_CHIRP, ofdm_symbols.flatten(), chirp.END_CHIRP))
    return signal


def decode_data(data_qpsk_values: npt.NDArray[np.complex128],signal: npt.NDArray[np.float64], plot: bool = False) -> None:
    aligned_signal = chirp.synchronise(signal, plot_correlations=plot)
    recv_ofdm_symbols = np.reshape(
        aligned_signal[chirp.START_CHIRP.size : -chirp.END_CHIRP.size], (-1, OFDM_SYMBOL_LENGTH)
    )

    if plot:
        fig, ax = plt.subplots()

        f, t_spec, Sxx = scipy.signal.spectrogram(aligned_signal, FS)

        pcm = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading="gouraud")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [sec]")
        cbar = fig.colorbar(pcm, ax=ax, label="Power/Frequency (dB/Hz)")

    num_ofdm_symbols = recv_ofdm_symbols.shape[0]
    length_ofdm_symbol = recv_ofdm_symbols.shape[1]
    

    assert num_ofdm_symbols % 2 == 1, f"Expected an odd number of total symbols"
    num_data_symbols = num_ofdm_symbols // 2
    num_pilot_symbols = num_data_symbols + 1
    pilot_qpsk_symbols = np.reshape(pilot.generate_pilot_blocks(num_pilot_symbols), (-1, DATA_BLOCK_LENGTH))
    known_qpsk_symbols = pilot.interleave_pilot_blocks(
        np.full((num_data_symbols, DATA_BLOCK_LENGTH), np.nan, dtype=np.complex128), pilot_qpsk_symbols
    )

    observed_frequency_gains = qpsk.decode_ofdm_symbol(recv_ofdm_symbols) / known_qpsk_symbols
    avg_gain = np.nanmean(observed_frequency_gains, axis=0)

    snr_estimates = estimate.estimate_snr(known_qpsk_symbols, qpsk.decode_ofdm_symbol(recv_ofdm_symbols), avg_gain)
    avg_snr = np.nanmean(snr_estimates, axis=0)

    

    if plot:
        fig, ax = plt.subplots()
        freqs = np.linspace(LOWER_FREQUENCY_BOUND, UPPER_FREQUENCY_BOUND, DATA_BLOCK_LENGTH, endpoint=False)
        for i, block_gain in enumerate(observed_frequency_gains[~np.isnan(observed_frequency_gains).any(axis=1)]):
            ax.plot(freqs, np.log10(np.abs(block_gain)), label=f"Block {i}")
        ax.plot(freqs, np.log10(np.abs(avg_gain)), label="Mean")
        ax.set_title("Frequency Gain Plot (in dB)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.legend()

    adjusted_data_qpsk_symbols = qpsk.wiener_filter(
        qpsk.decode_ofdm_symbol(recv_ofdm_symbols[1::2, :]), avg_gain, avg_snr
    )
    #################################################
    #reshpae into size 21 x1366
    data_qpsk_values = data_qpsk_values.reshape(-1, DATA_BLOCK_LENGTH)
    data_qpsk_values = data_qpsk_values[1::2,:]

    print(f"Adjusted data QPSK symbols shape: {adjusted_data_qpsk_symbols.shape}")
    print(f"Data QPSK values shape: {data_qpsk_values.shape}")
    print(f"average gain shape: {avg_gain.shape}")
    noise_var = estimate.estimate_noise_var(adjusted_data_qpsk_symbols,avg_gain,data_qpsk_values)
    llr_real, llr_imag = estimate.find_LLRs(adjusted_data_qpsk_symbols, avg_gain, noise_var)
    llr_real_flat = llr_real.flatten()
    llr_imag_flat = llr_imag.flatten()


    adjusted_llr_real = [llr_real_flat[i:i+LDPC_OUTPUT_LENGTH] for i in range(0, len(llr_real_flat), LDPC_OUTPUT_LENGTH)]
    adjusted_llr_imag = [llr_imag_flat[i:i+LDPC_OUTPUT_LENGTH] for i in range(0, len(llr_imag_flat), LDPC_OUTPUT_LENGTH)]
    if adjusted_llr_real[-1].size < LDPC_OUTPUT_LENGTH:
        adjusted_llr_real.pop()
    if adjusted_llr_imag[-1].size < LDPC_OUTPUT_LENGTH:
        adjusted_llr_imag.pop()

    decoded_llr_real = [ldpc_code.decode(chunk, ldpc_dec_type, corr_factor)[0] for chunk in adjusted_llr_real]
    decoded_llr_imag = [ldpc_code.decode(chunk, ldpc_dec_type, corr_factor)[0] for chunk in adjusted_llr_imag]
    # decode 1 if it is greater than 0 else decode 0
    decoded_llr_real = np.array(decoded_llr_real)
    decoded_llr_imag = np.array(decoded_llr_imag)
    decoded_llr_real = np.where(decoded_llr_real > 0, 1, 0)
    decoded_llr_imag = np.where(decoded_llr_imag > 0, 1, 0)

    # Combine the decoded real and imaginary parts
    decoded_bits = np.column_stack((decoded_llr_real, decoded_llr_imag)).flatten()
    ######################################
    print(f"Decoded LLRs: {decoded_bits[0:10]} values")
    print(f"Decoded LLRs shape: {decoded_bits.shape}")
    
    # TODO: remove
    rng = np.random.default_rng(seed=42)
    sent_data = bytes(rng.integers(256, size=BYTES_BLOCK_LENGTH * 200, dtype=np.uint8))
    data_qpsk_values = np.reshape(
        qpsk.qpsk_encode(np.unpackbits(np.frombuffer(sent_data, dtype=np.uint8))), (-1, DATA_BLOCK_LENGTH)
    )


    if plot:
        fig, axs = plt.subplots(2, 5)
        for sent_vals, recv_vals, ax in zip(data_qpsk_values, adjusted_data_qpsk_symbols, axs.flatten(), strict=True):
            positive_real_mask = np.real(sent_vals) > 0
            positive_imag_mask = np.imag(sent_vals) > 0
            mask_00 = positive_real_mask & positive_imag_mask
            mask_01 = (~positive_real_mask) & positive_imag_mask
            mask_11 = (~positive_real_mask) & (~positive_imag_mask)
            mask_10 = positive_real_mask & (~positive_imag_mask)

            for mask, bits in ((mask_00, "00"), (mask_01, "01"), (mask_10, "10"), (mask_11, "11")):
                ax.scatter(np.real(recv_vals[mask]), np.imag(recv_vals[mask]), label=bits)

            ax.legend()











if __name__ == "__main__":
    #rng = np.random.default_rng(seed=42)
    #data = bytes(rng.integers(256, size=BYTES_BLOCK_LENGTH * 200, dtype=np.uint8))
    #signal = encode_data(data)
    #wav.generate_wav("signal.wav", signal)
    def generate_test_data():
    # Generate random binary data
        rng = np.random.default_rng(seed=42)  # Use a fixed seed for reproducibility
        data = bytes(rng.integers(0, 256, size=BYTES_BLOCK_LENGTH, dtype=np.uint8))  # Example: 10 blocks of data

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
    
    recv_signal = wav.read_wav("2025-05-28_LT6.wav")
    decode_data(data_qpsk_values, recv_signal, False)
    print(f"shape of encoded_data_bits: {coded_data_bits.shape}")
    plt.show()
