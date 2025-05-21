import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from PIL import Image
import io

#Load the file to be transmitted
def load_file(filename):
    filepath = os.path.join("files", filename)
    
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")
    
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    
    file_size = len(file_bytes)  # in bytes
    return filepath, file_size, file_bytes

#Combine the bytes to be transmitted
def combine_bytes(filepath, file_size, file_bytes):
    combined_bytes = filepath.encode('ascii') + b'\0' + str(file_size).encode('ascii') + b'\0' + file_bytes
    return combined_bytes

#Create constellation
def qpsk_encode(bytes):
    bits = np.unpackbits(np.frombuffer(bytes, dtype=np.uint8))
    print(bits[:10])
    pad_len = int((-len(bits)) % (2*qpsk_block_length))
    bits = np.pad(bits, (0, pad_len), constant_values=0)

    bit_pairs = bits.reshape(-1, 2)
    gray_indices = ((bit_pairs[:, 0] << 1) | bit_pairs[:, 1])

    qpsk_constellation = np.array([
        (1+1j),   # 00
        (-1+1j),  # 01
        (1-1j),    # 10
        (-1-1j)  # 11
    ]) 

    qpsk_values = (qpsk_multiplier/ np.sqrt(2))*qpsk_constellation[gray_indices]
    return qpsk_values

#Add zeroes, conjugate and prefix
def create_signal(qpsk_values):
    qpsk_blocks=qpsk_values.reshape(-1, qpsk_block_length)
    zero_col = np.zeros((np.shape(qpsk_blocks)[0], 1), dtype=complex)
    conj_blocks = np.conj(np.fliplr(qpsk_blocks))
    X = np.hstack([zero_col, qpsk_blocks, zero_col, conj_blocks])
    x = np.fft.ifft(X, n=X_block_length)
    signal = np.hstack([x[:,-prefix_length:],x]).reshape(-1)
    return signal

#Combined function
def encode_file(filename):
    filepath, file_size, file_bytes = load_file(filename)
    print(file_bytes[:10])
    bytes = combine_bytes(filepath, file_size, file_bytes)
    print(bytes[:10])
    qpsk_values = qpsk_encode(bytes)
    print(qpsk_values[:10])
    signal = create_transmission(qpsk_values)
    print(signal[0:10])
    return signal

#for channel usage
def load_csv(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, filename)

    data=[]
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
    return np.array(data)

#use ft for linear convolution anyways
def fft_convolve(signal, h):
    n = signal.size + h.size - 1
    # next power-of-two for speed
    N = 1 << (n - 1).bit_length()
    # FFT, multiply, and inverse FFT
    y = np.fft.ifft(np.fft.fft(signal, N) * np.fft.fft(h, N))
    return np.real(y)[:len(signal)]

def channel_distortion(signal,h,noise_var,csv_name):
    signal = fft_convolve(signal, h)
    signal += np.random.normal(scale=np.sqrt(noise_var), size=np.shape(signal))
    print(signal[:10])
    np.savetxt(csv_name, signal, delimiter=',')

qpsk_multiplier = 30
X_block_length = 1024
prefix_length = 32
qpsk_block_length = int((X_block_length-2)/2)

h = load_csv('channel.csv')
noise_var = 0

filename = 'edith.wav'
csv_name = 'signal_1.csv'
signal = encode_file(filename)
channel_distortion(signal,h,noise_var,csv_name)

