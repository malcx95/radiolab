from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math


RAW_FILE = 'signal-malvi108.wav'

TOP_SIGNAL_FILTER_RANGE = [125000, 136000]
MIDDLE_SIGNAL_FILTER_RANGE = [80000, 110000]
LOWER_SIGNAL_FILTER_RANGE = [45000, 65000]

def band_pass_filter(data, order, start, end, sample_rate):
    b, a = signal.butter(order, [2 * start / sample_rate, 2 * end / sample_rate], 'bandpass', analog=False)
    return signal.lfilter(b, a, data)


def low_pass_filter(data, order, cutoff, sample_rate):
    # The Nyquist frequency is half the sample rate
    b, a = signal.butter(order, 2 * cutoff / sample_rate, 'low', analog=False) 
    return signal.lfilter(b, a, data)


def plot_filter(order, cutoff, sample_rate):
    # b, a = signal.butter(order, 2 * cutoff / sample_rate, 'low', analog=False)
    start, end = cutoff
    b, a = signal.butter(order, [2 * start / sample_rate, 2 * end / sample_rate], 'bandpass', analog=False)
    w, h = signal.freqz(b, a)
    plt.plot(0.5 * sample_rate * w/math.pi, np.abs(h), 'b')
    # plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    # plt.axvline(cutoff, color='k')
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.grid(which='both', axis='both')
    # plt.axvline(cutoff, color='green') # cutoff frequency
    plt.show()


def iq_demodulate(data, freq, bandwidth):
    pass


def low_pass_plot_transform(data, sample_rate, cutoff):
    num_samples = len(data)
    period = 1.0 / sample_rate

    new_data = low_pass_filter(data, 20, cutoff, sample_rate)

    transformed = fft(data)
    transformed_filtered = fft(new_data)

    xf = np.linspace(0.0, sample_rate // 2, num_samples // 2)
    plt.subplot(2, 1, 1)
    plt.plot(xf, 2.0 / num_samples * np.abs(transformed[0:num_samples // 2]), 'b')
    plt.grid()

    # plot the filtered transform
    plt.subplot(2, 1, 2)
    plt.plot(xf, 2.0 / num_samples * np.abs(transformed_filtered[0:num_samples // 2]), 'r')
    plt.grid()
    plt.show()


def band_pass_plot_transform(data, sample_rate, band):
    num_samples = len(data)
    period = 1.0 / sample_rate

    start, end = band
    new_data = band_pass_filter(data, 9, start, end, sample_rate)

    transformed = fft(data)
    transformed_filtered = fft(new_data)

    xf = np.linspace(0.0, sample_rate // 2, num_samples // 2)
    plt.subplot(2, 1, 1)
    plt.plot(xf, 2.0 / num_samples * np.abs(transformed[0:num_samples // 2]), 'b')
    plt.grid()

    # plot the filtered transform
    plt.subplot(2, 1, 2)
    plt.plot(xf, 2.0 / num_samples * np.abs(transformed_filtered[0:num_samples // 2]), 'r')
    plt.grid()
    plt.show()


def main():
    sample_rate, data = wavfile.read(RAW_FILE)
    # low_pass_plot_transform(data, sample_rate, 70000)
    band_pass_plot_transform(data, sample_rate, LOWER_SIGNAL_FILTER_RANGE)
    

if __name__ == "__main__":
    main()

