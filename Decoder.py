import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from PIL import Image
import io

#load 1 dim array of datta from column of csv file
def load_csv(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, filename)

    data=[]
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
    return np.array(data)

#Inverse filter and take useful portion
def inverse_filter(signal,h):
    y=signal.reshape(-1, 1056)[:,32:]
    Y = np.fft.fft(y, n=1024)
    H = np.fft.fft(h, n=1024)
    X = Y / np.where(np.abs(H) < 1e-6, 1e-6, H)
    values=X[:,1:512].reshape(-1)
    return values

#Decode qpsk constelation
def qpsk_decode(values):
    values = np.asarray(values)
    bits_real = (np.real(values) < 0).astype(int)
    bits_imag = (np.imag(values) < 0).astype(int)
    bits = np.column_stack((bits_imag, bits_real)).reshape(-1)
    bytes = np.packbits(bits).tobytes()
    return bytes

#Using the zero bytes, split the data into name, size and file
def extract_metadata(bytes):
    first_zero = bytes.find(b'\0')
    second_zero = bytes.find(b'\0', first_zero + 1)

    name = bytes[:first_zero].decode('ascii', errors='replace')
    size = int(bytes[first_zero+1:second_zero].decode('ascii', errors='replace'))
    file = bytes[second_zero+1:second_zero+1+size]
    return name, size, file

#Save decoded file next to code
def save_file(file, name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "wb") as f:
        f.write(file)

#Combined function
def decode_file(csvname):
    signal = load_csv(csvname)
    bytes = inverse_filter(signal,h)
    filename, file_size, file_bytes = extract_metadata(bytes)
    save_file(file_bytes, filename)

#Basic implementation for weekend task
h = load_csv('channel.csv')
for filenum in range(1,10):
    csvname = 'file'+str(filenum)+'.csv'
    decode_file(csvname)
