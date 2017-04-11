from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math


RAW_FILE = 'signal-malvi108.wav'


def data_filter(data, order, cutoff, ftype):
    b, a = signal.butter(order, cutoff, ftype, analog=True) 
    return signal.lfilter(b, a, data)


def custom_filter(data, start, end):
    transformed = fft(data)


def plot_filter(order, cutoff, sample_rate):
    b, a = signal.butter(order, cutoff, 'bandpass', analog=True)
    w, h = signal.freqs(b, a)
    plt.semilogx(w, abs(h))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.grid(which='both', axis='both')
    # plt.axvline(cutoff, color='green') # cutoff frequency
    plt.show()


def iq_demodulate(data, freq, bandwidth):
    pass


def plot_transform(data, sample_rate):
    num_samples = len(data)
    period = 1.0 / sample_rate

    new_data = data_filter(data, 4, [0 * math.pi, 700000 * math.pi], 'bandpass')

    transformed = fft(data)

    xf = np.linspace(0.0, sample_rate // 2, num_samples // 2)
    plt.plot(xf, 2.0 / num_samples * np.abs(transformed[0:num_samples // 2]))
    plt.grid()
    plt.show()


def main():
    sample_rate, data = wavfile.read(RAW_FILE)
    plot_transform(data, sample_rate)
    # plot_filter(4, [50000, 70000], sample_rate)
    

if __name__ == "__main__":
    main()

