from scipy.signal import csd, welch, chirp
import scipy.io.wavfile
import time
import scipy.io.wavfile as wav
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib import colormaps
def generate_QPSK_symbols(bit_pairs):
    #generate QPSK symbols from bit pairs
    QPSK_symbols = np.array([QPSK_symbol_map[bit_pair][0] for bit_pair in bit_pairs])
    #conjugate thee symbols
    QPSK_symbols_conjugate = np.conjugate(QPSK_symbols)
    #remove the first symbol
    QPSK_symbols_conjugate = np.delete(QPSK_symbols_conjugate, 0)
    #flip so the first symbol is at the end
    QPSK_symbols_conjugate = np.flip(QPSK_symbols_conjugate)
    QPSK_symbols[0] = 0
    QPSK_symbols = np.append(QPSK_symbols, 0)
    #append the conjugate symbols to the original symbols
    QPSK_symbols_trans = np.concatenate((QPSK_symbols, QPSK_symbols_conjugate))
    #assert the the QPSK symbol k is conjugate to QPSK symbol N-k for all k 
    return QPSK_symbols_trans

Fs= 48000
Duration = 10
max_freq = Fs/2
block_size = 8192
cp_len  = block_size//4
num_blocks = 10
QPSK_symbol_map = {
    (0, 0): (1 + 1j, 'red'),
    (0, 1): (-1 + 1j, 'blue'),
    (1, 0): (1 - 1j, 'green'),
    (1, 1): (-1 - 1j, 'orange')
}

def generate_psuedo_random_signal(block_size, num_blocks):
    psuedorandom_signal  = []
    decoded_bit_pairs_signal = []
    for i in range(num_blocks):
        np.random.seed(i)
        binary_vector = [int(bit) for bit in np.random.randint(0, 2, block_size)]
        # Check if the length of the binary vector is odd
        if len(binary_vector) % 2 != 0:
            binary_vector.append(0)
        bit_pairs = [(binary_vector[i], binary_vector[i+1]) for i in range(0, len(binary_vector), 2)]

        QPSK_symbols_trans = generate_QPSK_symbols(bit_pairs)
        #transform QPSK_sybolms_trans back to bit pairs
        decoded_bit_pairs = []  # List to store the decoded bit pairs
        for symbol in QPSK_symbols_trans:
            # Find the bit pair corresponding to the QPSK symbol
            for bit_pair, value in QPSK_symbol_map.items():
                if np.isclose(symbol, value[0]):  # Check if the symbol matches the value in the map
                    decoded_bit_pairs.append(bit_pair)  # Append the bit pair to the list
                    break
        decoded_bit_pairs_signal.append(decoded_bit_pairs)  
        psuedorandom_signal.append(QPSK_symbols_trans)
    
        for k in range(1,int(len(QPSK_symbols_trans)//2 + 1)):
            assert QPSK_symbols_trans[k] == np.conjugate(QPSK_symbols_trans[len(QPSK_symbols_trans)-k]), f"QPSK symbol {k} is not conjugate to QPSK symbol {len(QPSK_symbols_trans)-k-1}"
        assert np.isreal(QPSK_symbols_trans[0])
        assert np.isreal(QPSK_symbols_trans[len(QPSK_symbols_trans)//2])
        
 
    

    return psuedorandom_signal, decoded_bit_pairs_signal, bit_pairs,QPSK_symbols_trans

psuedo_random_signal,decoded_bit_pairs_signal,bit_pairs, QPSK_symbols_trans = generate_psuedo_random_signal(10,1)


print(QPSK_symbols_trans)
"""
print(decoded_bit_pairs_signal)

print(psuedo_random_signal)


# Flatten the list of decoded bit pairs before passing to generate_QPSK_symbols
flat_decoded_bit_pairs = [bit_pair for sublist in decoded_bit_pairs_signal for bit_pair in sublist]
print(flat_decoded_bit_pairs)
QPSK_symbols2 = np.array([QPSK_symbol_map[bit_pair][0] for bit_pair in flat_decoded_bit_pairs])
print(f"The flat decoded bit pairs are :{flat_decoded_bit_pairs}")
print(f"The QPSK symbols mapped from flat_decoded_bit_pairs are :{QPSK_symbols2}")
print(f"the bit pairs are {bit_pairs}")


time_domain = np.fft.ifft(QPSK_symbols_trans)
assert np.allclose(time_domain.imag, 0, atol=1e-6), "IFFT output not real!"
time_domain = np.real(time_domain)

# Add cyclic prefix
cyclic_prefix = time_domain[-cp_len:]
tx_block = np.concatenate([cyclic_prefix, time_domain])

print(f"Transmit block length: {len(tx_block)} samples ({cp_len} CP + {block_size} data)")

# Plot all QPSK symbols with their assigned colors
plt.figure(figsize=(10, 6))
plt.scatter(QPSK_symbols.real, QPSK_symbols.imag, c=symbol_colors)  # Use 'c' for color mapping

# Add labels and grid
plt.title("QPSK Symbols with Colors Assigned to Bit Pairs")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True)
plt.show()

# Play the signal
"""