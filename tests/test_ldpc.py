from matplotlib import pyplot as plt
import numpy as np

from modem.constants import BYTES_BLOCK_LENGTH, LDPC_INPUT_LENGTH, ldpc_code
from modem import main, qpsk

def generate_test_data():
    # Generate random binary data
    rng = np.random.default_rng(seed=4)  # Use a fixed seed for reproducibility
    data = bytes(rng.integers(0, 256, size=BYTES_BLOCK_LENGTH * 5, dtype=np.uint8))  # Example: 10 blocks of data

    # Encode the data using the encode_data function
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    pad_len = int((-len(data_bits)) % LDPC_INPUT_LENGTH)
    data_bits = np.pad(data_bits, (0, pad_len), constant_values=0).reshape((-1, LDPC_INPUT_LENGTH))
    coded_data_bits = np.vstack([ldpc_code.encode(ldpc_block) for ldpc_block in data_bits])
    data_qpsk_values = qpsk.qpsk_encode(coded_data_bits.flatten())

    return data, data_qpsk_values, coded_data_bits

def test_full():
    data, data_qpsk_values, coded_data_bits = generate_test_data()

    signal = main.encode_data(data)
    rng = np.random.default_rng(seed=4)
    recv_signal = np.pad(signal, (4000, 4000)) + rng.normal(scale=0.0028, size=signal.size + 8000)
    decoded_bits = main.decode_data(data_qpsk_values, recv_signal, False)
    bit_err_rate_per_ldpc_block = np.mean(coded_data_bits != decoded_bits, axis=1)
    print(f"{bit_err_rate_per_ldpc_block = }")


if __name__ == "__main__":
    test_full()
    plt.show()

