import math

# Sampling rate
FS = 48000

# Energy of each constellation symbol
QPSK_MULTIPLIER = 30

# FFT size used for OFDM
FFT_BLOCK_LENGTH = 8192

# Number of QPSK symbols per OFDM symbol
# Not using 0 and middle (zeros) and upper half of frequencies (complex conjugates)
QPSK_BLOCK_LENGTH = (FFT_BLOCK_LENGTH - 2) // 2 #4095

CYCLIC_PREFIX_LENGTH = 2048
LOWER_FREQUENCY_BOUND = 1000
UPPER_FREQUENCY_BOUND = 9000
POSITIVE_LOWER_BIN = math.ceil(FFT_BLOCK_LENGTH*LOWER_FREQUENCY_BOUND/( FS)) # 170
POSITIVE_UPPER_BIN = math.floor(FFT_BLOCK_LENGTH*UPPER_FREQUENCY_BOUND/( FS)) + 1 # 1535

# Number of QPSK data symbols per OFDM symbol
DATA_BLOCK_LENGTH = POSITIVE_UPPER_BIN - POSITIVE_LOWER_BIN
BYTES_BLOCK_LENGTH = DATA_BLOCK_LENGTH // 4

OFDM_SYMBOL_LENGTH = CYCLIC_PREFIX_LENGTH + FFT_BLOCK_LENGTH

DATA_PER_PILOT = 1

ldpc_standard = '802.11n'
ldpc_rate = '1/2'
ldpc_z_val = 81
ldpc_ptype = 'A'
