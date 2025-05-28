import numpy as np
import numpy.typing as npt

from modem import chirp, pilot, qpsk, wav
from modem.constants import BYTES_BLOCK_LENGTH


def encode_data(data: bytes, wav_name: str) -> npt.NDArray[np.float64]:
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    data_qpsk_values = qpsk.qpsk_encode(data_bits)
    data_ofdm_symbols = qpsk.encode_ofdm_symbol(data_qpsk_values)

    num_ofdm_blocks = data_ofdm_symbols.shape[0]
    print(f"Data encoded into {num_ofdm_blocks} OFDM blocks")

    pilot_ofdm_symbols = pilot.generate_pilot_blocks(num_ofdm_blocks + 1)
    ofdm_symbols = pilot.interleave_pilot_blocks(data_ofdm_symbols,pilot_ofdm_symbols)

    signal = np.concatenate((chirp.START_CHIRP, ofdm_symbols.flatten(), chirp.END_CHIRP))
    wav.generate_wav(wav_name, signal)


def decode_data(signal: npt.NDArray[np.float64]):
    ...

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    data = bytes(rng.integers(256, size=BYTES_BLOCK_LENGTH * 200, dtype=np.uint8))
    encode_data(data, "signal.wav")
