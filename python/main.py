from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

RAW_FILE = 'signal-malvi108.wav'


def low_pass(data, order, cutoff):
    b, a = signal.butter(order, cutoff, 'lowpass')
    return signal.lfilter(b, a, data)


def plot_filter(order, cutoff):
    b, a = signal.butter(order, cutoff, 'lowpass', analog=True)
    w, h = signal.freqs(b, a)
    plt.semilogx(w, abs(h))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.grid(which='both', axis='both')
    plt.axvline(cutoff, color='green') # cutoff frequency
    plt.show()


def iq_demodulate(data, freq, bandwidth):
    pass


def plot_transform(data, sample_rate):
    num_samples = len(data)
    period = 1.0 / sample_rate

    transformed = fft(data)

    xf = np.linspace(0.0, sample_rate // 2, num_samples // 2)
    plt.plot(xf, 2.0 / num_samples * np.abs(transformed[0:num_samples // 2]))
    plt.grid()
    plt.show()


def main():
    sample_rate, data = wavfile.read(RAW_FILE)
    #plot_transform(data, sample_rate)
    plot_filter(4, 1000)
    

if __name__ == "__main__":
    main()

