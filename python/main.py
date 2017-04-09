from scipy.fftpack import fft
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

RAW_FILE = 'signal-malvi108.wav'

def main():
    sample_rate, data = wavfile.read(RAW_FILE)
    num_samples = len(data)
    period = 1.0 / sample_rate

    transformed = fft(data)

    xf = np.linspace(0.0, sample_rate // 2, num_samples // 2)
    plt.plot(xf, 2.0 / num_samples * np.abs(transformed[0:num_samples // 2]))
    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    main()

