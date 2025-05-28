import numpy as np
import numpy.typing as npt

from modem.constants import (
    CYCLIC_PREFIX_LENGTH,
    FFT_BLOCK_LENGTH,
    QPSK_BLOCK_LENGTH,
    QPSK_MULTIPLIER,
    LOWER_FREQUENCY_BOUND,
    UPPER_FREQUENCY_BOUND,
    POSITIVE_LOWER_BIN,
    POSITIVE_UPPER_BIN,
    DATA_BLOCK_LENGTH,
)
print("hello from qpsk.py")

def qpsk_encode(bytes: bytes) -> npt.NDArray[np.complex128]:
    """Encode bytes into constellation symbols
    using QPSK in the frequency domain, before OFDM.

    Parameters
    ----------
    bytes : bytes
        Bytes with data to be encoded.
        Bytes will be padded to full OFDM symbol,
        but must fit within one symbol.

    Returns
    ----------
    qpsk_values : npt.NDArray[np.complex128]
        Returns 1D array of QPSK constellation symbols of size QPSK_BLOCK_LENGTH.
    """
    bits = np.unpackbits(np.frombuffer(bytes, dtype=np.uint8))
    pad_len = int((-len(bits)) % (2 * DATA_BLOCK_LENGTH))
    bits = np.pad(bits, (0, pad_len), constant_values=0) # FOR FUTURE : pad with non zeroez to test.

    bit_pairs = bits.reshape(-1, 2)
    gray_indices = (bit_pairs[:, 0] << 1) | bit_pairs[:, 1]

    qpsk_constellation = np.array(
        [
            (1 + 1j),  # 00
            (-1 + 1j),  # 01
            (1 - 1j),  # 10
            (-1 - 1j),  # 11
        ]
    )

    qpsk_values = (QPSK_MULTIPLIER / np.sqrt(2)) * qpsk_constellation[gray_indices]
    return qpsk_values


def qpsk_decode(values: npt.NDArray[np.complex128],) -> bytes:
    """Decode received constellation symbols
    in the frequency domain into bytes using QPSK"""
    values = np.asarray(values)
    bits_real = (np.real(values) < 0).astype(int)
    bits_imag = (np.imag(values) < 0).astype(int)
    bits = np.column_stack((bits_imag, bits_real)).reshape(-1)
    bytes = np.packbits(bits).tobytes()
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
    into time-domain OFDM symbol for transmitti"""

    #Input: qpsk_values is a 1D array of QPSK symbols (1+j,1-j,-1+j,-1-j...1+j,-1-j,1+j,-1-j) 
    # Output : Multiple arrays of QPSK symbols, each of with qpsk data of size QPSK_BLOCK_LENGTH_DATA and 
    #          zeroes of length POSITIVE_LOWER_BIN and POSITIVE_UPPER_BIN at the beginning and end of the array.
    assert qpsk_values.shape[0] % DATA_BLOCK_LENGTH == 0, f"QPSK values need to be an even multiple of OFDM length, got {qpsk_values.shape[0]}"
    qpsk_data_blocks = qpsk_values.reshape(-1, DATA_BLOCK_LENGTH)
    qpsk_blocks = np.zeros((qpsk_data_blocks.shape[0], FFT_BLOCK_LENGTH//2), dtype=np.complex128)
    qpsk_blocks[:, POSITIVE_LOWER_BIN:POSITIVE_UPPER_BIN] = qpsk_data_blocks
    qpsk_blocks = qpsk_blocks[:, 1:]

    conj_blocks = np.conj(np.fliplr(qpsk_blocks))
    zero_col = np.zeros((np.shape(qpsk_blocks)[0], 1), dtype=np.complex128)
    X = np.hstack([zero_col, qpsk_blocks, zero_col, conj_blocks])

    x = np.fft.ifft(X, n=FFT_BLOCK_LENGTH)
    x = np.hstack([x[:, -CYCLIC_PREFIX_LENGTH:], x])
    np.testing.assert_allclose(np.imag(x), np.zeros_like(x), atol=1e-14)

    return np.real(x)

    chunk_size = QPSK_BLOCK_LENGTH_DATA
    qpsk_values_chunks = [qpsk_values[i:i + chunk_size] for i in range(0, len(qpsk_values), chunk_size)]
    padded_chunks = []
    for chunk in qpsk_values_chunks:
        qpsk_blocks = np.pad(chunk, 
        (POSITIVE_LOWER_BIN, QPSK_BLOCK_LENGTH -len(chunk) - POSITIVE_LOWER_BIN), 
        constant_values=0 )

        padded_chunks.append(qpsk_blocks)

    qpsk_blocks = np.array(padded_chunks)
    qpsk_blocks = qpsk_blocks.reshape(-1, QPSK_BLOCK_LENGTH)

    conj_blocks = np.conj(np.fliplr(qpsk_blocks))
    X = np.hstack([zero_col, qpsk_blocks, zero_col, conj_blocks])
    for i in range(X.shape[0]):
        for k in range(1, int(len(X[i]) // 2 + 1)):
            assert X[i][k] == np.conjugate(X[i][len(X[i]) - k]), (
                f"QPSK symbol {k} is not conjugate to QPSK symbol {len(X) - k - 1}"
            )
            assert np.isreal(X[i][0])
            assert np.isreal(X[i][len(X) // 2]) 
    x = np.fft.ifft(X, n=FFT_BLOCK_LENGTH)
    signal = np.hstack([x[:, -CYCLIC_PREFIX_LENGTH:], x]).reshape(-1)
    np.testing.assert_allclose(np.imag(signal), np.zeros_like(signal), atol=1e-14)
    
    print("passes test")
    return np.real(signal), X

# encode_ofdm_symbol(random_symbols)

def decode_ofdm_symbol(
    ofdm_symbol: npt.NDArray[np.float64], channel_gains: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Decode time-domain OFDM symbol into
    constellation symbols in frequency-domain"""
    ofdm_symbol = ofdm_symbol[:, CYCLIC_PREFIX_LENGTH:]  # Discard cyclic prefix
    # Ignore bits 0 and 512 (zeros) and upper half of frequencies (complex conjugates)
    freq_values = np.fft.fft(ofdm_symbol)
    freq_values = freq_values[:, POSITIVE_LOWER_BIN:POSITIVE_UPPER_BIN]
    return freq_values / channel_gains

#####################test the encode qpsk function

"""

# Define the QPSK constellation symbols
qpsk_symbols = np.array([1+1j, 1-1j, -1-1j, -1+1j], dtype=np.complex128)

# Generate a random array of 12,000 QPSK symbols
random_qpsk_symbols = np.random.choice(qpsk_symbols, size=12000)

# Test the encode_ofdm_symbol function
signal, X = encode_ofdm_symbol(random_qpsk_symbols)

"""