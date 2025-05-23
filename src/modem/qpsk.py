import numpy as np
import numpy.typing as npt

from .constants import (
    FFT_BLOCK_LENGTH,
    QPSK_BLOCK_LENGTH,
    QPSK_MULTIPLIER,
    CYCLIC_PREFIX_LENGTH,
)


def qpsk_encode(bytes: bytes) -> npt.NDArray[np.complex128]:
    """Encode bytes into constellation symbols
    using QPSK in the frequency domain, before OFDM.
    Bytes will be padded to full OFDM symbol,
    but must fit within one symbol"""
    max_bytes = (2 * QPSK_BLOCK_LENGTH) // 8
    assert len(bytes) <= max_bytes, f"Cannot send more than {max_bytes} bytes"
    bits = np.unpackbits(np.frombuffer(bytes, dtype=np.uint8))
    pad_len = int((-len(bits)) % (2 * QPSK_BLOCK_LENGTH))
    bits = np.pad(bits, (0, pad_len), constant_values=0)

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


def qpsk_decode(values: npt.NDArray[np.complex128]) -> bytes:
    """Decode received constellation symbols
    in the frequency domain into bytes using QPSK"""
    values = np.asarray(values)
    bits_real = (np.real(values) < 0).astype(int)
    bits_imag = (np.imag(values) < 0).astype(int)
    bits = np.column_stack((bits_imag, bits_real)).reshape(-1)
    bytes = np.packbits(bits).tobytes()
    return bytes


def encode_ofdm_symbol(
    qpsk_values: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    """Encode constellation symbols in frequency-domain
    into time-domain OFDM symbol for transmitting"""
    qpsk_blocks = qpsk_values.reshape(-1, QPSK_BLOCK_LENGTH)
    zero_col = np.zeros((np.shape(qpsk_blocks)[0], 1), dtype=complex)
    conj_blocks = np.conj(np.fliplr(qpsk_blocks))
    X = np.hstack([zero_col, qpsk_blocks, zero_col, conj_blocks])
    x = np.fft.ifft(X, n=FFT_BLOCK_LENGTH)
    signal = np.hstack([x[:, -CYCLIC_PREFIX_LENGTH:], x]).reshape(-1)
    np.testing.assert_allclose(np.imag(signal), np.zeros_like(signal), atol=1e-14)
    return np.real(signal)


def decode_ofdm_symbol(
    ofdm_symbol: npt.NDArray[np.float64], channel_gains: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Decode time-domain OFDM symbol into
    constellation symbols in frequency-domain"""
    ofdm_symbol = ofdm_symbol[CYCLIC_PREFIX_LENGTH:]  # Discard cyclic prefix
    freq_values = np.fft.fft(ofdm_symbol)

    freq_values = freq_values / channel_gains

    # Ignore bits 0 and 512 (zeros) and upper half of frequencies (complex conjugates)
    return freq_values[1:FFT_BLOCK_LENGTH//2]
