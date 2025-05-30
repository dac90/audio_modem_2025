import numpy as np
import numpy.typing as npt
import time

from modem import chirp, pilot, qpsk, wav
from modem.constants import BYTES_BLOCK_LENGTH
from modem.chirp import synchronise   ###
from modem.qpsk import recieve_signal ###


### = Ido's addition 

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
    return signal, ofdm_symbols # ido's addition

### Ido's implementation
def decode_data(signal: npt.NDArray[np.float64], ofdm_symbols: npt.NDArray[np.float64]) -> None:
    recv_signal = recieve_signal()
    aligned_recv_signal = chirp.synchronise(recv_signal, sum(map(len, ofdm_symbols)), plot_correlations=True)
    recv_ofdm_symbols = np.split(aligned_recv_signal[chirp.START_CHIRP.size:-chirp.END_CHIRP.size], len(ofdm_symbols))


    ...

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    data = bytes(rng.integers(256, size=BYTES_BLOCK_LENGTH * 200, dtype=np.uint8))
    encode_data(data, "signal.wav")
