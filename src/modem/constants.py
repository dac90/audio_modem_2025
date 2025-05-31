import math
from modem.ldpc import code
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
UPPER_FREQUENCY_BOUND = 16000
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
ldpc_dec_type = 'sumprod2'
corr_factor = 0.7 # should have a discussion about this value.
ldpc_code = code(standard=ldpc_standard, rate=ldpc_rate, z=ldpc_z_val, ptype=ldpc_ptype)
LDPC_INPUT_LENGTH = ldpc_code.K #972 currently
LDPC_OUTPUT_LENGTH = ldpc_code.Nv #972 * 2 = 1944  # Rate 1/2

