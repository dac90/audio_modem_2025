import os
import numpy as np
from modem import qpsk
from modem.constants import FFT_BLOCK_LENGTH
import pytest


@pytest.mark.parametrize("num_bytes", (2, 20, 200, 1024))
def test_qpsk(num_bytes: int):
    data = os.urandom(num_bytes)
    channel_gain = np.ones(FFT_BLOCK_LENGTH)

    qpsk_symbols = qpsk.qpsk_encode(data)
    print(qpsk_symbols.shape)
    ofdm_symbols = qpsk.encode_ofdm_symbol(qpsk_symbols)
    print(ofdm_symbols.shape)

    recv_qpsk_symbols = qpsk.decode_ofdm_symbol(ofdm_symbols, channel_gain)
    print(recv_qpsk_symbols.shape)
    recv_bytes = qpsk.qpsk_decode(recv_qpsk_symbols)

    recv_data = recv_bytes[: len(data)]  # Truncate padding
    assert data == recv_data, f"Received {recv_data}, expected {data}"
