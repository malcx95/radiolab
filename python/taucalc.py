from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy import signal
import os
import matplotlib.pyplot as plt
import numpy as np
import math

W_F1 = 141500
W_F2 = 141501

SAMPLE_LENGTH = 150000

def get_x_transformed(length, sample_rate):
    w1 = 2 * np.pi * W_F1
    w2 = 2 * np.pi * W_F2
    return fft([0.001 * (np.cos(w1 * t / sample_rate) + 
                         np.cos(w2 * t / sample_rate)) 
                for t in range(length)])


def get_h_transformed(y_transformed, length, sample_rate):
    result = []
    x = get_x_transformed(length, sample_rate)
    assert len(x) == len(y_transformed)
    for i in range(len(y_transformed)):
        result.append(np.abs(y_transformed[i] / x[i]) if x[i] != 0 else float('nan'))
    return result

def calculate_h_transformed(f1, f2):
    return [np.abs(1 + 0.9 * np.exp(-1j * 2 * np.pi * f1 * tau)) +
            np.abs(1 + 0.9 * np.exp(-1j * 2 * np.pi * f2 * tau))
            for tau in range(500)]

def correlation_function(data, sample_rate):
    result = []
    
    dt = 1 / sample_rate

    dtau = sample_rate // 1000
    x_signal = data[:SAMPLE_LENGTH]

    for tau in range(500):
        sum_ = 0
        for i in range(len(x_signal)):
            sum_ += x_signal[i] * data[i + tau * dtau] * dt
        result.append(sum_)

    return np.array(result)

def plot_correlation(correlation):
    plt.plot(correlation, 'b')
    plt.title("Correlation")
    plt.grid()
    plt.show()

