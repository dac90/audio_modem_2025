import numpy as np
import numpy.typing as npt
import time

from modem import chirp, pilot, qpsk, wav
from modem.constants import (BYTES_BLOCK_LENGTH,DATA_PER_PILOT,DATA_BLOCK_LENGTH,
                            OFDM_SYMBOL_LENGTH, ldpc_standard, ldpc_rate, ldpc_z_val, ldpc_ptype) ###
from modem.chirp import synchronise   ###-
from modem.qpsk import recieve_signal ###
from modem.ldpc import code ###


### = Ido's addition 
ldpc_code = code(standard=ldpc_standard, rate=ldpc_rate, z=ldpc_z_val, ptype=ldpc_ptype)
proto = ldpc_code.assign_proto()
pcmat = ldpc_code.pcmat()

def encode_data(data: bytes, wav_name: str) -> npt.NDArray[np.float64]:
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    data_bits = ldpc_code.encode(data_bits)  ###
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
    aligned_recv_signal = chirp.synchronise(recv_signal, sum(map(len, ofdm_symbols)), plot_correlations=False)
    RECIEVED_BLOCKS = int(len(aligned_recv_signal[chirp.START_CHIRP.size:-chirp.END_CHIRP.size])/OFDM_SYMBOL_LENGTH)
    recv_ofdm_symbols = np.reshape(aligned_recv_signal[chirp.START_CHIRP.size:-chirp.END_CHIRP.size], (RECIEVED_BLOCKS, -1))
    RECIEVED_PILOT_BLOCKS = int(1+((RECIEVED_BLOCKS-1)//(1+DATA_PER_PILOT)))
    RECIEVED_DATA_BLOCKS = RECIEVED_BLOCKS - RECIEVED_PILOT_BLOCKS
    data_qpsk_values = np.reshape(data_qpsk_values,[-1,DATA_BLOCK_LENGTH])
    pilot_qpsk_symbols = np.reshape(pilot_qpsk_symbols,[-1,DATA_BLOCK_LENGTH])
    known_qpsk_symbols = pilot.interleave_pilot_blocks(np.full((RECIEVED_DATA_BLOCKS, DATA_BLOCK_LENGTH), np.nan, dtype=np.complex128), pilot_qpsk_symbols)
    ...

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    data = bytes(rng.integers(256, size=BYTES_BLOCK_LENGTH * 200, dtype=np.uint8))
    encode_data(data, "signal.wav")
