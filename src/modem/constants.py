# Sampling rate
from typing import Literal
import numpy as np


FS = 48000

# Energy of each constellation symbol
QPSK_MULTIPLIER = 30

# FFT size used for OFDM
FFT_BLOCK_LENGTH = 8196

# Number of QPSK symbols per OFDM symbol
# Not using 0 and middle (zeros) and upper half of frequencies (complex conjugates)
QPSK_BLOCK_LENGTH = (FFT_BLOCK_LENGTH-2)//2

CYCLIC_PREFIX_LENGTH = 2048

OFDM_SYMBOL_LENGTH = CYCLIC_PREFIX_LENGTH + QPSK_BLOCK_LENGTH
