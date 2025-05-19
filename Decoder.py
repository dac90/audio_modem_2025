import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from PIL import Image
import io

def load_csv(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, filename)

    data=[]
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
    return np.array(data)

def qpsk_decode(values):
    values = np.asarray(values)
    bits_real = (np.real(values) < 0).astype(int)
    bits_imag = (np.imag(values) < 0).astype(int)
    return np.column_stack((bits_imag, bits_real)).reshape(-1)

def signal_to_binary(signal,h):
    y=signal.reshape(-1, 1056)[:,32:]
    Y = np.fft.fft(y, n=1024)
    H = np.fft.fft(h, n=1024)

    X = Y / np.where(np.abs(H) < 1e-6, 1e-6, H)
    values=X[:,1:512].reshape(-1)
    binary=qpsk_decode(values)
    return binary

def extract_metadata(bits):
    bytes = np.packbits(bits).tobytes()

    first_zero = bytes.find(b'\0')
    second_zero = bytes.find(b'\0', first_zero + 1)

    name = bytes[:first_zero].decode('ascii', errors='replace')
    size = int(bytes[first_zero+1:second_zero].decode('ascii', errors='replace'))
    file = bytes[second_zero+1:second_zero+1+size]
    return name, size, file

def save_file(file, name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "wb") as f:
        f.write(file)


h = load_csv('channel.csv')
for filenum in range(1,10):
    filename = 'file'+str(filenum)+'.csv'
    signal = load_csv(filename)
    bits = signal_to_binary(signal,h)
    name, size, file = extract_metadata(bits)
    save_file(file, name)
