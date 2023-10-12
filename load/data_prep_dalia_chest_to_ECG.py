import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
import scipy.signal
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq

sos = signal.butter(4, [0.7, 40], btype='bandpass', fs=700, output='sos')
sos_ppg = signal.butter(4, [0.5, 4], btype='bandpass', fs=64, output='sos')
sos_imu = signal.butter(4, [0.8, 15], btype='bandpass', fs=700, output='sos')
signal_list_imu = []
signal_list_ecg = []
label_list, bpm_label_list, activity_list = [], [], []
ppg_fft_list = []

## MOD Minxuan: add parameter for different sampling rate
original_fs = 700
target_fs = 80
window_size = 8 # seconds
window_offset = 2 # seconds

for file in os.listdir():
    d = os.path.join('', file)
    if os.path.isdir(d) and d != '.idea':
        print(d)
        with open(d + '/' + d + '.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            ppg_signal = data['signal']['wrist']['BVP']
            ecg_signal_orig = data['signal']['chest']['ECG']
            chest_IMU_signal = data['signal']['chest']['ACC']
            wrist_IMU_signal = data['signal']['wrist']['ACC']
            chest_IMU_signal = chest_IMU_signal[:,2] # only use z-axis
            ###
            filtered_imu = signal.sosfilt(sos, chest_IMU_signal)
            segments = np.lib.stride_tricks.sliding_window_view(np.squeeze(filtered_imu), original_fs * window_size)[::original_fs * window_offset]  # 8 seconds
            z_scored = stats.zscore(segments, axis=1)
            chest_IMU_signal = scipy.signal.resample(z_scored, target_fs * window_size, axis=1)  # 8 Seconds of 80 Hz
            ###
            filtered_ecg = signal.sosfilt(sos, ecg_signal_orig)
            segments = np.lib.stride_tricks.sliding_window_view(np.squeeze(filtered_ecg), original_fs * window_size)[::original_fs * window_offset]  # 8 seconds
            z_scored = stats.zscore(segments, axis=1)
            ecg_signal = scipy.signal.resample(z_scored, target_fs * window_size, axis=1)  # 8 Seconds of 80 Hz
            ###
            '''
            filtered_ppg = signal.sosfilt(sos, ppg_signal)
            segments = np.lib.stride_tricks.sliding_window_view(np.squeeze(filtered_ppg), 64 * 8)[::128]
            z_scored = stats.zscore(segments, axis=1)
            resampled = scipy.signal.resample(z_scored, 200, axis=1)  # 8 Seconds of 25 Hz
            '''
            ###
            segments_activity = np.lib.stride_tricks.sliding_window_view(data['activity'].squeeze(), 4 * 8)[::8]
            activity_list.append(segments_activity[:,0])
            ###

            ## MOD: Minxuan, add windowing for rpeaks and then manually convert it
            labels = data['rpeaks']
            ###
            orig_time = np.arange(ecg_signal_orig.size)
            orig_time[labels] = -1
            orig_time[orig_time != -1] = 0
            orig_time[orig_time == -1] = 5

            ## Approach 2: convert the peak position manually
            convert_rate = target_fs / original_fs
            orig_time_segments = np.lib.stride_tricks.sliding_window_view(np.squeeze(orig_time), original_fs * window_size)[::original_fs * window_offset]
            convert_pos_0, convert_pos_1 = np.where(orig_time_segments == 5)[0], np.where(orig_time_segments == 5)[1]

            pos_0 = (convert_pos_0).astype(int)
            pos_1 = (convert_pos_1 * convert_rate).astype(int)

            down_sample_time = np.zeros(ecg_signal.shape)
            down_sample_time[pos_0, pos_1] = 5

            label_list.append(down_sample_time) # append here, not expand_dims
            ###
            bpm_labels = data['label']
            signal_list_imu.append(chest_IMU_signal)
            signal_list_ecg.append(ecg_signal)
            # label_list.append(np.expand_dims(labels, 1))
            bpm_label_list.append(bpm_labels)

dataDalia = dict(data_ecg=signal_list_ecg, data_imu=signal_list_imu, data_rpeaks=label_list, data_bpm_labels=bpm_label_list, data_activity=activity_list)
with open('dataDaliaMod.pkl', 'wb') as handle:
    pickle.dump(dataDalia, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('exit')