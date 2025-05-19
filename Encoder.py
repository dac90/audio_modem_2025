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
    return filepath.encode('ascii') + b'\0' + file_size.encode('ascii') + b'\0' + file_bytes

#Create constellation
def qpsk_encode():

#Combined function
def encode_file(filename):
    