import numpy as np
from modem.constants import DATA_PER_PILOT, RNG_SEED, QPSK_BLOCK_LENGTH
from modem import qpsk


def generate_pilot_blocks(num: int):
    rng = np.random.default_rng(seed=RNG_SEED) # creates a random number generator with a fixed seed for reproducibility
    # creates pilot bits (0,1) the total number of bits is equal to the size parameter.
    pilot_bits = rng.integers(2, size=QPSK_BLOCK_LENGTH * 2 * num, dtype=np.uint8)
    pilot_qpsk_symbols = qpsk.qpsk_encode(pilot_bits) # Turn pilot bits into QPSK symbols
    return pilot_qpsk_symbols.reshape(-1, QPSK_BLOCK_LENGTH)


def interleave_pilot_blocks(data_blocks, pilot_blocks):
    result = []
    data_index = 0

    for i in range(len(pilot_blocks)):
        # Add a pilot block
        result.append(pilot_blocks[i])
        
        # Add data blocks
        if i < len(pilot_blocks):
            for _ in range(DATA_PER_PILOT):
                if data_index < len(data_blocks):
                    result.append(data_blocks[data_index])
                    data_index += 1
    return np.array(result)

def extract_pilot_blocks(joint_blocks: np.ndarray):
    step = 1 + DATA_PER_PILOT
    pilot_mask = np.arange(len(joint_blocks)) % step == 0

    pilot_blocks = joint_blocks[pilot_mask]
    data_blocks = joint_blocks[~pilot_mask]

    return data_blocks, pilot_blocks
