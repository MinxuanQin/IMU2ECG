import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pywt
from scipy import signal

'''
This file contains some helper functions for data extraction
and visualization
'''
## General question: how to close used items to save memory?
## Define some constants
window_size = 8 # seconds
window_offset = 2 # seconds

save_path = '/cluster/scratch/minqin/sp_imu/IMU2ECG/plots/'

def load_pickle_file(path):
    '''
    load dataset, read fs from signal
    fs = int, ecg_signal.second_dim / 8 (8 seconds as window size)
    it contains 15 subjects, each item as an array of a list
    '''
    dataset = np.load(path, allow_pickle=True)

    ## keywords from data extraction
    ecg_signals = dataset['data_ecg']
    imu_signals = dataset['data_imu']

    rpeaks = dataset['data_rpeaks']
    bpm_labels = dataset['data_bpm_labels']
    activity_labels = dataset['data_activity']

    ## get sampling rate

    fs = int(ecg_signals[0].shape[1] / window_size)
    return fs, ecg_signals, imu_signals, rpeaks, bpm_labels, activity_labels

def spec_analysis(ecg_signal, imu_signal, fs_spec, title, save_path):
    '''
    fs_spec must not be fs, depends on the concrete case
    plot ecg and imu signal spectrogram, title and saving path given by the caller
    '''
    ## compute spec
    f_ecg, t_ecg, Sxx_ecg = signal.spectrogram(ecg_signal, fs=fs_spec, window='hann', nperseg=fs_spec, noverlap=fs_spec/2)
    f_imu, t_imu, Sxx_imu = signal.spectrogram(imu_signal, fs=fs_spec, window='hann', nperseg=fs_spec, noverlap=fs_spec/2)
    
    ## layout
    s_min, s_max = np.min(np.concatenate((Sxx_ecg, Sxx_imu))), np.max(np.concatenate((Sxx_ecg, Sxx_imu)))
    fig = plt.figure(figsize=(30, 10))
    
    plt.suptitle(title, fontsize=20)
    fig1 = plt.subplot(121)
    c_ecg = plt.pcolormesh(t_ecg, f_ecg, Sxx_ecg, vmin=s_min, vmax=s_max, cmap='Reds')
    plt.colorbar(c_ecg)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.title('ECG spectrogram')

    fig2 = plt.subplot(122)
    c_imu = plt.pcolormesh(t_imu, f_imu, Sxx_imu, vmin=s_min, vmax=s_max, cmap='Reds')
    plt.colorbar(c_imu)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.title('IMU spectrogram')
    
    plt.savefig(save_path)
    plt.close(fig1)
    plt.close(fig2)

def plot_ecg_imu(ecg_signal, imu_signal, title, save_path):
    '''
    plot ecg and imu signal, title and saving path given by the caller
    '''
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    axs[0].plot(ecg_signal)
    axs[1].plot(imu_signal)
    axs[0].set_title(title)
    plt.savefig(save_path)
    plt.close(fig)

def plot_wt_transforms(ecg_signal, imu_signal, signal, coefficient, is_inverse, title, save_path):
    '''
    plot wavelet transform details coefficients
    ecg_signal: ground truth ecg signal
    imu_signal: original "wrist vibration signal", in this dataset from one channel from chest IMU
    signal: original imu signal
    coefficient: result from pywt, here only details coefficients
    is_inverse = True: coefficient is from level x to 1
    '''

    levels = len(coefficient)
    offset = 2
    fig, axs = plt.subplots((levels+offset), 1, figsize=(20, 10))
    plt.suptitle(title, fontsize=20)

    ## plot original signal
    ## Another version: ecg alone in one axs
    axs[1].plot(ecg_signal, 'r', label='ECG')
    axs[1].legend()
    axs[0].plot(imu_signal, 'r', label='BCG')
    axs[0].plot(signal, label='noise')
    axs[0].legend()
    for i in range(levels):
        if is_inverse:
            axs[i+offset].plot(coefficient[levels-offset-i])
        else:
            axs[i+offset].plot(coefficient[i])
        axs[i+offset].set_ylabel('Level {}'.format(i+1))

    plt.savefig(save_path)
    plt.close(fig)

def plot_baseline3_res(signal, detail_coeff, short_energy, local_max_idx, step2_idx, step3_idx, title, save_path):
    '''
    plot baseline3 results (for a single entry)
    signal: original signal
    detail_coeff: swt details coefficients from baseline 3
    short_energy: short time energy from baseline3
    local_max_idx: local maxima idx of short time energy from baseline3
    step2_idx: result from baseline3, step 2, marks the noise idx
    step3_idx: result from baseline3, step 3, marks the noise idx
    title, save_path: similar to above
    '''
    levels = len(detail_coeff)
    fig, axs = plt.subplots((levels + 1), 1, figsize=(20, 10))
    plt.suptitle(title, fontsize=20)

    ## plot original signal
    axs[0].plot(signal)
    for i in range(levels):
        # axs[i+1].plot(detail_coeff[i], color='y')
        axs[i+1].plot(short_energy[i], color='b')
        axs[i+1].plot(local_max_idx[i], short_energy[i][local_max_idx[i]], 'go')
        axs[i+1].plot(step2_idx[i], short_energy[i][step2_idx[i]], 'ro')
        axs[i+1].plot(step3_idx[i], short_energy[i][step3_idx[i]], 'ys')
        axs[i+1].set_ylabel('Level {}'.format(i+1))

    plt.savefig(save_path)
    # plt.show()
    plt.close(fig)