import numpy as np

from .constants import QPSK_BLOCK_LENGTH, QPSK_MULTIPLIER


def qpsk_encode(bytes: bytes) -> np.ndarray:
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


def qpsk_decode(values: np.ndarray) -> bytes:
    values = np.asarray(values)
    bits_real = (np.real(values) < 0).astype(int)
    bits_imag = (np.imag(values) < 0).astype(int)
    bits = np.column_stack((bits_imag, bits_real)).reshape(-1)
    bytes = np.packbits(bits).tobytes()
    return bytes
