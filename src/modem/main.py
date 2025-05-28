import numpy as np
import numpy.typing as npt

from modem import chirp, qpsk, wav
from modem.constants import BYTES_BLOCK_LENGTH


def encode_data(data: bytes, wav_name: str):
    qpsk_values = qpsk.qpsk_encode(data)
    print(f"{qpsk_values.shape = }")
    ofdm_symbols = qpsk.encode_ofdm_symbol(qpsk_values)

    signal = np.concatenate((chirp.START_CHIRP, ofdm_symbols.flatten(), chirp.END_CHIRP))

    wav.generate_wav(wav_name, signal)


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    data = bytes(rng.integers(256, size=BYTES_BLOCK_LENGTH * 200, dtype=np.uint8))
    encode_data(data, "signal.wav")
