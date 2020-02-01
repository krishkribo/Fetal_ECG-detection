import numpy as np
import pywt
from matplotlib import pyplot as plt
from padasip.filters import FilterLMS
from pywt import Wavelet
from scipy import signal

from ssnf import ssnf

DATA_FREQUENCY = 1000
MAX_FIGURE_RANGE = 5000

# High-pass filter parameters
BUTTER_FILTER_ORDER = 2
BUTTER_CRITICAL_FREQUENCY = 600 / DATA_FREQUENCY

# LMS parameters
LMS_STEP_SIZE = 0.01
LMS_FILTER_LENGTH = 6

# Wavelet transform parameters
WAVELET_STYLE = Wavelet("bior1.5")


def fprint(s: str):
    print("-----> " + s)


# Read data from file
def get_data(file_name: str):
    with open(file_name + '.txt') as f:
        return [float(x) for x in f.readlines()]


# Plotting
def plot_single_data(data: [float], title: str = ''):
    r = min(MAX_FIGURE_RANGE, len(data))
    plt.figure()
    plt.plot(list(range(r)), data[:r])
    if title:
        plt.title(title)
    plt.draw()


def plot_data(data: [[float]], title: str, t_lst: [str]):
    r = min(MAX_FIGURE_RANGE, len(data[0]))
    fig, ax = plt.subplots(np.shape(data)[0])
    plt.subplots_adjust(hspace=1)
    fig.suptitle(title)
    for i in range(0, np.shape(data)[0]):
        ax[i].plot([x for x in range(r)], data[i][:r])
        ax[i].set_title(t_lst[i])
    plt.draw()


def subplot_data(data: [[float]], title: str):
    sub_titles = ["Filtered with thorax wavelet coefficients :" + str(i) for i in range(0, np.shape(data)[0])]
    plot_data(data, title, sub_titles)


def subplot_data1(data: [[float]], title: str):
    r = min(MAX_FIGURE_RANGE, len(data[0]))
    fig, ax = plt.subplots(np.shape(data)[0])
    plt.subplots_adjust(hspace=1)
    fig.suptitle(title)
    d = 1
    for i in range(0, np.shape(data)[0]):
        ax[i].plot([x for x in range(0, len(data[i]))], data[i])
        j = 1 if i % 2 == 0 or i == 0 else 2
        ax[i].set_title("Abdome signal:" + str(d) + " Thorax signal:" + str(j))
        if j == 2:
            d += 1
    plt.draw()


# High-pass Butter filter
def hp_filter(inp_signal):
    sos = signal.butter(BUTTER_FILTER_ORDER, BUTTER_CRITICAL_FREQUENCY, btype='highpass', fs=DATA_FREQUENCY,
                        output='sos')
    return signal.sosfilt(sos, inp_signal)


# Normalize input data
def normalize_data(data: [float]):
    mean = sum(data) / len(data)
    data = [d - mean for d in data]
    m = max(data)
    return [d / m for d in data]


# Stationary wavelet transform
def swt(data: [[float]]):
    return pywt.swt(data, wavelet=WAVELET_STYLE, level=5, trim_approx=True)


# Inverse stationary wavelet transform
def inv_swt(coeff):
    return pywt.iswt(coeff, wavelet=WAVELET_STYLE)


# Least Mean Squares algorithm
def apply_lms(desired_value: [float], input_matrix: [[float]]):
    y, _, _ = FilterLMS(LMS_FILTER_LENGTH, mu=LMS_STEP_SIZE, w='random').run(desired_value, input_matrix)
    return y


def calculate_lms(input_signals: [[float]], reference_signals: [[float]]):
    res = []
    inp_data = np.transpose(input_signals)
    for ref_data in reference_signals:
        ref_data = np.transpose(ref_data)
        y = [apply_lms(ref_data[:, i], inp_data) for i in range(0, np.shape(ref_data)[1])]
        res.append(y)
    return res


def pre_processing(data: [str]):
    data = [hp_filter(d) for d in data]
    # plot_data(hp_data_filtered,"High pass filtered data",inp_data)
    fprint("High pass filter applied")
    return [normalize_data(d) for d in data]


if __name__ == "__main__":
    files = ['abdomen1', 'abdomen2', 'abdomen3', 'thorax1', 'thorax2']
    # Load data from file
    data = [get_data(f) for f in files]
    plot_data(data, "Raw data", files)
    data = pre_processing(data)
    subplot_data(data, "Pre-processing")

    # step 1 - process the signal by the stationary wavelet transfrom method
    data = [swt(d) for d in data]
    fprint("Data transformed into wavelet domain")

    # step 2 - filter the wavelet coefficients obtained from the previous step
    wavelet_data = []
    for abdomen in range(3):
        # send abdomen data as the input signal and thorax signals as the reference
        wavelet_data.append(calculate_lms(data[abdomen], data[3:]))
        for thorax in range(2):
            subplot_data(wavelet_data[abdomen][thorax],
                         "Input data: Abdomen signal {}, Reference data: thorax_signal {}".format(abdomen + 1,
                                                                                                  thorax + 1))
        fprint("Abdomen {} data filtered".format(abdomen))
    data = wavelet_data

    # Step 3: SSNF filteration
    ssnf_data = []
    for abdomen in range(3):
        ssnf_data.append([])
        for thorax in range(2):
            ssnf_data[abdomen].append(ssnf(data[abdomen][thorax], 5, 5 * [10]))
            subplot_data(ssnf_data[abdomen][thorax],
                         "SSNF applied Input data: Abdomen signal {}, Reference data: thorax_signal {}".format(
                             abdomen + 1, thorax + 1))

    # Step 4: Inverse wavelet
    n_data = np.zeros((np.shape(data)[2], np.shape(data)[3]))
    count = 0
    for i in range(len(data)):
        for j in range(len(data[1])):
            inv_wav = inv_swt(data[i][j])
            n_data[count] = inv_wav
            count = count + 1

    subplot_data1(n_data, "inverse wavelet transformed data")
    fprint("Inverse wavelet transform applied")

    plot_data([ssnf_data[2][0][4], ssnf_data[2][1][4]],
              "Abdomen signal 3, with thorax reference data",
              ['Thorax 1', 'Thorax 2'])

    plt.show()
