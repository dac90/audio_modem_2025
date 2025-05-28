import numpy as np
from modem.constants import DATA_BLOCK_LENGTH, DATA_PER_PILOT
from modem import qpsk

def generate_pilot_blocks(num: int):
    rng = np.random.default_rng(seed=42)
    pilot_bits = rng.integers(2, size=DATA_BLOCK_LENGTH * 2 * num)
    pilot_qpsk_symbols = qpsk.qpsk_encode(pilot_bits)
    pilot_ofdm_symbols = qpsk.encode_ofdm_symbol(pilot_qpsk_symbols)
    return pilot_ofdm_symbols


def interleave_pilot_blocks(data_blocks, pilot_blocks):
    result = []
    data_index = 0

    for i in range(len(pilot_blocks)):
        # Add a pilot block
        result.append(pilot_blocks[i])
        
        # Add data blocks if it's not the last pilot
        if i < len(pilot_blocks) - 1:
            for _ in range(DATA_PER_PILOT):
                if data_index < len(data_blocks):
                    result.append(data_blocks[data_index])
                    data_index += 1
    return np.array(result)
