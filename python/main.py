from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy import signal
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import taucalc

RAW_FILE = 'signal-malvi108.wav'

OUTPUT_DIR = 'output'

I_OUTPUT = 'i_signal.wav'
Q_OUTPUT = 'q_signal.wav'

TOP_SIGNAL_FILTER_RANGE = [125000, 136000]
MIDDLE_SIGNAL_FILTER_RANGE = [88000, 100000]
LOWER_SIGNAL_FILTER_RANGE = [45000, 65000]
W_RANGE = [139000, 144000]

DELTA = 0.24

W_F1 = 141500
W_F2 = 141501
W_FILTER_ORDER = 8

# tau2 - tau1 in ms
TIME_DELAY = 0.420

ECHO_STRENGTH = 0.9


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


def iq_demodulate(data, band, delta, sample_rate):
    start, end = band
    center_freq = (start + end) / 2
    bandwidth = end - start

    print("Removing echo...")
    no_echo = remove_echo(data, sample_rate)

    print("Filtering out the interesting band...")

    filtered_data = band_pass_filter(no_echo, 10, start, end, sample_rate)

    print("Multiplying x(t) with 2cos(2pi*fc*t + delta)...")

    x_cos = []
    for i in range(len(filtered_data)):
        x_cos.append(2 * filtered_data[i] * np.cos(2 * np.pi * center_freq * i / sample_rate + delta))

    print("Multiplying x(t) with 2sin(2pi*fc*t + delta)...")

    x_sin = []
    for i in range(len(filtered_data)):
        x_sin.append(-2 * filtered_data[i] * np.sin(2 * np.pi * center_freq * i / sample_rate + delta))

    print("Low pass filtering the xi(t) component...")
    i_comp = low_pass_filter(np.array(x_cos), 10, bandwidth / 2, sample_rate)

    print("Low pass filtering the xq(t) component...")
    q_comp = low_pass_filter(np.array(x_sin), 10, bandwidth / 2, sample_rate)

    print("Done demodulating.")

    return i_comp, q_comp
        

def low_pass_plot_transform(data, sample_rate, cutoff):
    num_samples = len(data)

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


def band_pass_plot_transform(data, sample_rate, band, order=9):
    num_samples = len(data)

    start, end = band
    new_data = band_pass_filter(data, order, start, end, sample_rate)

    b, a = signal.butter(order, [2 * start / sample_rate, 2 * end / sample_rate], 'bandpass', analog=False)
    w, h = signal.freqz(b, a)

    transformed = fft(data)
    transformed_filtered = fft(new_data)

    xf = np.linspace(0.0, sample_rate // 2, num_samples // 2)
    plt.subplot(2, 2, 1)
    plt.plot(xf, 2.0 / num_samples * np.abs(transformed[0:num_samples // 2]), 'b')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(0.5 * sample_rate * w/math.pi, np.abs(h), 'g')
    plt.grid()

    # plot the filtered transform
    plt.subplot(2, 2, 3)
    plt.plot(xf, 2.0 / num_samples * np.abs(transformed_filtered[0:num_samples // 2]), 'r')
    plt.grid()
    plt.show()



def normalize(data):
    """
    Scales all elements in the data so they are all between -1 and 1.
    """
    data_max = max([abs(x) for x in data])
    return np.array([x / data_max for x in data])


def plot_iq_signals(i_data, q_data, sample_rate):
    num_samples = len(i_data)

    i_transformed = fft(i_data)
    q_transformed = fft(q_data)

    xf = np.linspace(0.0, sample_rate // 2, num_samples // 2)

    plt.subplot(2, 2, 1)
    plt.plot(i_data, 'b')
    plt.title("xI(t)")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(q_data, 'r')
    plt.title("xQ(t)")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(xf, 2.0 / num_samples * np.abs(i_transformed[0:num_samples // 2]), 'b')
    plt.title("|XI(f)|")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xf, 2.0 / num_samples * np.abs(q_transformed[0:num_samples // 2]), 'r')
    plt.title("|XQ(f)|")
    plt.grid()
    plt.show()


def iq_demodulate_different_deltas(data, sample_rate):
    for n in range(10):

        d = n / 20
        i, q = iq_demodulate(data, MIDDLE_SIGNAL_FILTER_RANGE, d * np.pi, sample_rate)

        print("Normalizing...")
        i_norm = normalize(i)
        q_norm = normalize(q)

        print("Writing output files...")
        write_file(os.path.join(OUTPUT_DIR, str(d).replace('.', '_') + I_OUTPUT), sample_rate, i_norm)
        write_file(os.path.join(OUTPUT_DIR, str(d).replace('.', '_') + Q_OUTPUT), sample_rate, q_norm)
        print("Done.")


def remove_echo(data, sample_rate):
    """
    Removes echo from the data.
    """
    res = list(data)
    delay_in_samples = int(TIME_DELAY * sample_rate)

    for i in range(len(data) - delay_in_samples):
        res[i + delay_in_samples] -= ECHO_STRENGTH * res[i]
    return np.array(res)


def iq_demodulate_single(data, sample_rate, delta):
    i, q = iq_demodulate(data, MIDDLE_SIGNAL_FILTER_RANGE, delta * np.pi, sample_rate)

    print("Normalizing...")
    i_norm = normalize(i)
    q_norm = normalize(q)

    print("Writing output files...")
    write_file(os.path.join(OUTPUT_DIR, I_OUTPUT), sample_rate, i_norm)
    write_file(os.path.join(OUTPUT_DIR, Q_OUTPUT), sample_rate, q_norm)

    plot_iq_signals(i_norm, q_norm, sample_rate)


def plot_w(data, sample_rate):
    filtered_data = band_pass_filter(data, 7, *W_RANGE, sample_rate)
    envelope = np.abs(signal.hilbert(filtered_data))
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.plot(filtered_data, 'b')
    plt.plot(envelope, 'g')
    plt.grid()
    plt.show()
    

def write_file(file_name, sample_rate, data):
    decimated = []
    for i in range(0, len(data), 10):
        decimated.append(data[i])
    wavfile.write(file_name, sample_rate // 10, np.array(decimated))


def main():
    sample_rate, data = wavfile.read(RAW_FILE)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # noise = band_pass_filter(data, 10, *LOWER_SIGNAL_FILTER_RANGE, sample_rate)

    # correlation = taucalc.correlation_function(noise, sample_rate)
    # taucalc.plot_correlation(correlation)
    
    # y = band_pass_filter(data, W_FILTER_ORDER, *W_RANGE, sample_rate)
    # y_transformed = fft(y)

    # h_transformed = taucalc.get_h_transformed(y_transformed, len(y), sample_rate)

    # print("H(f1) = {}, H(f2) = {}".format(h_transformed[W_F1], h_transformed[W_F2]))
    # print("H(0) = {}".format(h_transformed[0]))

    # xf = np.linspace(0.0, sample_rate // 2, len(y) // 2)
    # plt.plot(xf, 2.0 / len(y) * np.abs(h_transformed[0:len(y) // 2]), 'b')

    # H = taucalc.calculate_h_transformed(W_F1, W_F2)
    # print(H)
    
    iq_demodulate_single(data, sample_rate, 0.24)

    # correlation_function(data, sample_rate)
    # band_pass_plot_transform(data, sample_rate, W_RANGE, order=8)

    # low_pass_plot_transform(data, sample_rate, 70000)
    # band_pass_plot_transform(data, sample_rate, MIDDLE_SIGNAL_FILTER_RANGE)
    

if __name__ == "__main__":
    main()

