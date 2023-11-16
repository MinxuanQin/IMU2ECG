import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pywt
import scipy
from tqdm import tqdm
import os

## Defined by Minxuan
from help_tool import load_pickle_file, spec_analysis
from help_tool import plot_ecg_imu, plot_wt_transforms, plot_baseline3_res

## Modified by Minxuan, NLMS with adaptive step size
from adaptive_filter.lms import NLMS

## Define some constants
window_size = 8 # seconds
window_offset = 2 # seconds

## window for short time energy
window_ms = 200 # ms

filter_name = 'db4'
level = 6

## For step 5, adaptive filter
filter_order = 10
f_0 = 0.7
p = 0.1

plot_dir = 'plots/w2f7n10_fix/'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

def swt_decomposition(imu_signals, filter, level):
    '''
    imu_signals: 2d array
    level: int
    return: coeff (list of tuples, smooth coeff and detail coeff)
    details coeff (2d array, sequence inversed [that means from level 1 to x])
    '''
    ## swt works for 2d array, for plotting purpose, we need manually combine needed coefficients from the result
    ## swt returns a list of tuples, each represents one level
    ## shape of one level tuple entry is imu_signals.shape
    coeff_imu = pywt.swt(imu_signals, filter, level=level)

    ## we only need detail coefficients for the step 2 & 3
    ## arrange a list of all detail coefficients in each level
    swt_details_list = [coeff[1] for coeff in coeff_imu]
    return coeff_imu, swt_details_list

def compute_short_energy(signal, fs, window_ms=100):
    '''
    signal: 2d signal, here coeffs at level x
    fs: sampling rate, here obtained as constants
    window_ms: compute energy within this window, in ms
    return: short energy of the signal, shape (signal.shape[0], signal.shape[1] / act_window)
    '''
    ## compute active window size
    act_window = int(window_ms * fs / 1000)

    ## Meeting 13.10.2023: No need to make it as sliding window
    res = []
    for l in range(signal.shape[0]):
        short_energy = []
        for i in range(0, signal.shape[1], act_window):
            energy = np.sum(np.square(signal[l][i:i+act_window]))
            short_energy.append(energy)
        res.append(short_energy)
    return np.array(res)

def swt_step2(swt_details_list, fs, window_ms=100):
    '''
    swt_details_list: list of 2d array, each array is the detail coeff at one level
    fs: sampling rate, here obtained as constants
    window_ms: compute energy within this window, in ms
    return: short energy, local maximum (P_k) and idx in short time energy, idx classified as "noise"
    '''
    ## compute short energy for each level
    ## use scipy.signal.argrelextrema to find local maxima

    short_energy_list = []
    local_max_list = []
    local_max_idx_list = []
    noise_idx_list = []
    for level in swt_details_list:
        short_time_energy = compute_short_energy(level, fs, window_ms=window_ms)
        short_energy_list.append(short_time_energy)

        ## find local maxima, store them in a list (because each element has different items)
        single_local_max_list = []
        single_local_max_idx_list = []
        single_noise_idx_list = []
        for single_short_energy in short_time_energy:
            ## local maxima
            local_max_idx = scipy.signal.argrelextrema(single_short_energy, np.greater)
            single_local_max_idx = local_max_idx[0]
            single_local_max = single_short_energy[single_local_max_idx]

            single_local_max_idx_list.append(single_local_max_idx)
            single_local_max_list.append(single_local_max)

            ## mean, std of local maxima of one element
            ## TODO: For one element or for all? Here compute for each entry
            single_local_max_mean = np.mean(single_local_max)
            single_local_max_std = np.std(single_local_max)

            ## noise
            single_noise_idx_list1 = []
            single_short_thres = single_local_max_mean + 3 * single_local_max_std
            for single_idx, single_local_max in zip(single_local_max_idx, single_local_max):
                if single_local_max > single_short_thres:
                    single_noise_idx_list1.append(single_idx)
            
            single_noise_idx_list.append(single_noise_idx_list1)

        
        local_max_list.append(single_local_max_list)
        local_max_idx_list.append(single_local_max_idx_list)
        noise_idx_list.append(single_noise_idx_list)
    return short_energy_list, local_max_list, local_max_idx_list, noise_idx_list

def swt_step3(swt_details_list, local_max_idx_list):
    '''
    swt_details_list: list of 2d array, each array is the detail coeff at one level
    local_max_idx_list: from step2, 6 levels, each level local max of short time energy

    return: idx classified as "noise" in step 3 [in short time energy]
    '''
    ## compute interval for each level
    swt_local_interval = []
    for level in range(len(swt_details_list)):
        single_level_interval = [l_local_idx[1:] - l_local_idx[:-1] for l_local_idx in local_max_idx_list[level]]
        swt_local_interval.append(single_level_interval)
    ## median of cardio activive interval for all subjects is 3.0 (but only for window_ms = 100)

    swt_noise_step3_idx = []
    for level in range(len(swt_details_list)):
        single_level_noise_idx = []
        single_level_interval = swt_local_interval[level]

        for i in range(len(single_level_interval)):
            single_entry_noise_idx = []
            for j in range(len(single_level_interval[i]) - 1):
                if (single_level_interval[i][j] > 3.0) and (single_level_interval[i][j+1] > 3.0):
                    continue
                else:
                    single_entry_noise_idx.append(local_max_idx_list[level][i][j+1])
            single_level_noise_idx.append(single_entry_noise_idx)
        swt_noise_step3_idx.append(single_level_noise_idx)
    return swt_noise_step3_idx

def swt_step4(swt_coeff, swt_details_list, local_max_idx, step2_idx, step3_idx, fs, window_ms=100):
    '''
    swt_coeff: tuple of smooth coeff and detail coeff, from step 1
    step2_idx: idx classified as "noise" in step 2
    step3_idx: idx classified as "noise" in step 3
    return: swt_details_list of the reference noise, reference noise 2d signal
    '''

    convert_rate = fs * (window_ms / 1000)
    swt_step4_detail_masks = []
    swt_step4_idx = []
    ## mask for each level
    for level in range(len(swt_details_list)):
        single_level_step2 = step2_idx[level]
        single_level_step3 = step3_idx[level]
        single_level_step4 = []
        i = 0
        single_level_mask = np.zeros_like(swt_details_list[0])
        for local_max, step2, step3 in zip(local_max_idx[level], single_level_step2, single_level_step3):
            single_entry_step4 = [local_idx for local_idx in local_max if local_idx in step2 or local_idx in step3]
            single_level_step4.append(single_entry_step4)
            for ele in single_entry_step4:
                single_level_mask[i, int((ele-1)*convert_rate): int(ele*convert_rate)] = 1
            i = i + 1
        swt_step4_detail_masks.append(single_level_mask)
        swt_step4_idx.append(single_level_step4)

    ## reference noise generation
    coeff_step4 = swt_coeff.copy()
    for idx, level in enumerate(coeff_step4):
        tmp = list(level)
        tmp[1] = swt_step4_detail_masks[idx] * tmp[1]
        coeff_step4[idx] = tuple(tmp)
    return swt_step4_idx, coeff_step4

def compute_acf(imu_signals):
    '''          
    Compute ACF of given signal
    imu_signals: 2d array, each row one signal
    return: acf values stored in a 2d array, shape (num_signals, ks)
    '''
    ## shape: (num_signals,)
    c_0 = np.std(imu_signals, axis=1)
    ## shape: (num_signals, 1)
    y_bar = np.mean(imu_signals, axis=1, keepdims=True)
    T = imu_signals.shape[1]
    acf_res = []
    for k in range(T):
        if k == 0:
            y_t = imu_signals - y_bar
            y_tk = imu_signals - y_bar
        else:
            y_t = imu_signals[:,:-k] - y_bar
            y_tk = imu_signals[:,k:] - y_bar
        
        ## shape: (num_signals,)
        c_k = []
        for ele_y_t, ele_y_tk in zip(y_t, y_tk):
            ## each ele is in 1d
            c_k.append(np.sum(ele_y_t * ele_y_tk))
        c_k = np.array(c_k) / T
        r_k = c_k / (c_0 ** 2)
        acf_res.append(r_k)
    return np.array(acf_res).T

def swt_adaptive_std_est(signals, fs):
    '''
    Estimate heart rate via ACF, compute std from last half cycle, but not used in step5
    signals: 2d array, each row is one signal
    return: 1d array, std estimated from each row, estimated bpm (for baseline 1)
    '''
    acf_functions = compute_acf(signals)
    ## Find local maxima between 0.4 and 1.5 seconds (prior knowledge)
    ## store bpm result
    res_bpm = []
    res_std = []

    for (acf, signal) in zip(acf_functions, signals):
        local_max_idx = scipy.signal.argrelextrema(acf, np.greater)[0]
        low_thres = 0.4 * fs
        high_thres = 1.5 * fs
        for idx in local_max_idx:
            if idx < low_thres or idx > high_thres:
                continue
            else:
                break
        ## convert it into seconds & bpm
        hr_sec = idx / fs
        hr_bpm = 60 / hr_sec
        res_bpm.append(hr_bpm)

        ## compute std, use the avarage value of all segments
        segment_len = int(fs * hr_sec)
        split_signal = np.array_split(signal, segment_len)
        segment_std = []
        for seg in split_signal:
            ## last half cycle
            segment_std.append(np.std(seg[segment_len//2:]))
        res_std.append(np.mean(np.array(segment_std)))

    return np.array(res_std), np.array(res_bpm)

def swt_step5(signals, ref_noises, fs, filter_order, f_0, p):
    '''
    signals: 2d array, each row is one IMU signal
    ref_noises: 2d array, each row is one reference noise from step 4
    fs: int, sampling rate
    filter_order: int, filter order for adaptive filter
    f_0: float, preset value for step size
    p: float, scaler term for std
    return: 2d array, each row is one filtered signal by NLMS adaptive filter
    return: error vectors and output vectors for visualization
    '''
    ## Get the hrv estimation from ACF Function
    _, bpm_signals = swt_adaptive_std_est(signals, fs)
    ## convert to sampling rate
    hrv_signals = bpm_signals / 60 * fs
    hrv_signals = hrv_signals.astype(int)
    '''
    fig = plt.figure(figsize=(20, 10))
    plt.hist(hrv_signals, bins=80)
    plt.savefig(plot_dir + 'hrv_est.png')
    plt.close(fig)
    '''

    res = []
    error_vec = []
    output_vec = []
    ## New idea: update the coefficient through the whole process
    filter_coeff_size = filter_order + 1
    init_coef = np.random.randn(filter_coeff_size)
    # sub_nlms = NLMS(step=0.1, filter_order=filter_order, gamma=1e-12, init_coef=init_coef, f_0=f_0, p=p, hr_est=hrv_est)
    with tqdm(total=len(signals)) as pbar:
        pbar.set_description('Step 5:')
        for sig, ref, hrv_est in zip(signals, ref_noises, hrv_signals):
            #init_coef = np.ones(filter_order+1)
            curr_nlms = NLMS(step=0.1, filter_order=filter_order, gamma=1e-12, init_coef=init_coef, f_0=f_0, p=p, hr_est=60)
            ## Change it because of the new idea
            curr_nlms.fit(ref, sig)
            adaptive_noise = curr_nlms.output_vector.real
            filtered_signal = sig - adaptive_noise

            ## update init_coef for next step
            init_coef = curr_nlms.coef_vector[-1]

            res.append(filtered_signal)
            error_vec.append(curr_nlms.error_vector.real)
            output_vec.append(curr_nlms.output_vector.real)
            pbar.update(1)
    return np.array(res), np.array(error_vec), np.array(output_vec)
    
def swt_peak_through(signals, r_peaks):
    '''
    signals: 2d array, each row is one signal (here signal after step 5)
    r_peaks: 2d array, each row gound truth of r peaks
    returns 4 features of the peaks and throughs with idx and label
    '''
    sig_features = []
    
    for (idx_number, single_sig) in enumerate(signals):
        # sig_feature = []
        ## find local maxima & minima
        local_max_idx = scipy.signal.argrelextrema(single_sig, np.greater)[0]
        local_min_idx = scipy.signal.argrelextrema(single_sig, np.less)[0]

        ## find the first local extrema and the last
        if local_max_idx[0] < local_min_idx[0]:
            local_max_idx = local_max_idx[1:]
        if local_max_idx[-1] > local_min_idx[-1]:
            local_max_idx = local_max_idx[:-1]
        
        ## find the corresponding values
        local_max = single_sig[local_max_idx]
        local_min = single_sig[local_min_idx]

        ## concat two terms: Now we have through-peak=through-peak-through-...
        local_ex_idx = np.sort(np.concatenate((local_max_idx, local_min_idx)))
        local_ex_sort_idx = np.argsort(np.concatenate((local_max_idx, local_min_idx)))
        local_ex = np.concatenate((local_max, local_min))[local_ex_sort_idx]
        ## compute a-h, use math library
        for i in range(1, len(local_ex)-1):
            ## single_feature includes idx_number, spike idx, r_peak_label, 4 selected features
            single_feauture = []
            single_feauture.append(idx_number)

            x_1, y_1 = local_ex_idx[i-1], local_ex[i-1]
            x_2, y_2 = local_ex_idx[i], local_ex[i]
            x_3, y_3 = local_ex_idx[i+1], local_ex[i+1]

            single_feauture.append(x_2)

            p1 = np.array([x_1, y_1])
            p2 = np.array([x_2, y_2])
            p3 = np.array([x_3, y_3])
            
            a = np.linalg.norm(p2-p1)
            b = np.linalg.norm(p3-p2)
            c_prime = np.linalg.norm(p3-p1)

            c = np.abs(x_3-x_1)
            d = np.abs(y_2-y_1)
            e = np.abs(y_2-y_3)

            f = np.arccos((b*b-a*a-c_prime*c_prime)/(-2*a*c_prime))
            #g = np.arccos((a*a-b*b-c_prime*c_prime)/(-2*b*c_prime))
            #h = np.arccos((c_prime*c_prime-a*a-b*b)/(-2*a*b))

            ## labeling of features, here not consider robustness
            if r_peaks[idx_number][x_2] == 0:
                seg_label = 0
            else:
                seg_label = 1
            
            single_feauture.append(seg_label)
            single_feauture.append(a+b)
            #single_feauture.append(c)
            single_feauture.append(d)
            single_feauture.append(e)
            single_feauture.append(f)
            #single_feauture.append(g)
            #single_feauture.append(h)
            sig_features.append(single_feauture)
    return np.array(sig_features)

    
def main():
    fs, ecg_signals, imu_signals, rpeaks, bpm_labels, act_labels = load_pickle_file('DaliaDataset/dataDaliaMod.pkl')
    #dataset = np.load('dataDalia.pkl', allow_pickle=True)
    #rpeak_labels = np.load('aligned_rpeaks.pkl', allow_pickle=True)
    #imu_signals = dataset['data_imu']
    #ecg_signals = dataset['data_ecg']
    #act_labels = dataset['data_activity']
    #fs = 80

    is_plot = True
    plot_num = 50

    ## for one subject
    test_sub_idx = 0
    ## filter out transit signals
    ## TODO: filter out transit signals
    coeff_imus, swt_details_list = swt_decomposition(imu_signals[test_sub_idx], filter_name, level)
    short_energy_list, local_max_list, local_max_idx_list, noise_idx_list = swt_step2(swt_details_list, fs, window_ms=window_ms)
    ## step 3
    swt_noise_step3_idx = swt_step3(swt_details_list, local_max_idx_list)
    ## step 4
    swt_step4_noise_idx, swt_noise_coeff = swt_step4(coeff_imus, swt_details_list, local_max_idx_list, noise_idx_list, swt_noise_step3_idx, fs, window_ms=window_ms)

    ref_noise = pywt.iswt(swt_noise_coeff, filter_name)
    ## test for subject 0, ten indeces
    test_idx = np.random.randint(0, len(swt_details_list[test_sub_idx]), plot_num)

    #test_idx = [55, 263, 1790, 1857, 2225, 2260, 2372, 2498, 2735, 2918]
    ## test if acf makes sense
    '''
    acf_functions = compute_acf(imu_signals[test_sub_idx])
    for i in test_idx:
        test_imu = imu_signals[test_sub_idx][i]
        test_acf = acf_functions[i]
        #plt.figure(figsize=(20,10))
        #plt.plot(test_acf)
        #plt.savefig(plot_dir + 'acf_{}.png'.format(i))
        ## trials on improvement of acf_hrv
        ## baseline 1: prior knowledge of ecg signal
        test_local_max_idx = scipy.signal.argrelextrema(test_acf, np.greater)[0]
        low_thres = 0.4 * fs
        high_thres = 1.5 * fs
        for j in test_local_max_idx:
            if j < low_thres or j > high_thres:
                continue
            else:
                print("ACF idx {} local max: ".format(i), j)
                break
    '''
    ## have a sense of "sigma product"
    '''
    sigma_bcg, bpm_bcg = swt_adaptive_std_est(imu_signals[test_sub_idx], fs)
    sigma_noise, _ = swt_adaptive_std_est(ref_noise, fs)

    
    ## plot histogram of sigmas
    ## have a look at sigma product to determine p and f_0
    sigma_product = sigma_bcg * sigma_noise
    plt.figure(figsize=(20,10))
    plt.hist(sigma_bcg, bins=30, alpha=0.5, label='BCG')
    plt.hist(sigma_noise, bins=30, alpha=0.5, label='Ref Noise')
    #plt.hist(sigma_product, bins=50, label='sigma product')
    plt.title('Baseline 3, Histogram of std of BCG and Ref Noise')
    plt.legend()
    plt.savefig(plot_dir + 'hist_sigma_bcg_noise.png')
    '''
    ## step 5, adaptive filter
    swt_step5_filtered, swt_step5_error, swt_step5_output = swt_step5(imu_signals[test_sub_idx], ref_noise, fs, filter_order, f_0, p)

    ### title with window size
    if is_plot:
        for idx in test_idx:
            step3_title_idx = 'Baseline 3, Step 3, Idx {}, window size {} ms'.format(idx, window_ms)
            step3_plot_dir = plot_dir + 'step3_subject0_idx_{}.png'.format(idx)
            title_ref_noise = 'Baseline 3, Step 4, Idx {}, window size {} ms'.format(idx, window_ms)
            step4_plot_dir = plot_dir + 'step4_subject0_idx_{}.png'.format(idx)
            plot_detail = []
            plot_short_energy = []
            plot_local_max_idx = []
            plot_noise_idx = []
            plot_step3_idx = []
            plot_step4_coeff = []

            for i in range(len(swt_details_list)):
                plot_detail.append(swt_details_list[i][idx])
                plot_short_energy.append(short_energy_list[i][idx])
                plot_local_max_idx.append(local_max_idx_list[i][idx])
                plot_noise_idx.append(noise_idx_list[i][idx])
                plot_step3_idx.append(swt_noise_step3_idx[i][idx])
                ## step 4 details coefficient
                plot_step4_coeff.append(swt_noise_coeff[i][1][idx])
            plot_baseline3_res(imu_signals[test_sub_idx][idx], plot_detail, plot_short_energy, plot_local_max_idx, plot_noise_idx, plot_step3_idx, step3_title_idx, step3_plot_dir)
            plot_wt_transforms(ecg_signals[test_sub_idx][idx], imu_signals[test_sub_idx][idx], ref_noise[idx], plot_step4_coeff, True, title_ref_noise, step4_plot_dir)

            step5_title = 'Baseline 3, Step 5, Idx {}, Act label {}, window size {} ms'.format(idx, act_labels[test_sub_idx][idx], window_ms)
            step5_plot_dir = plot_dir + 'step5_subject0_idx_{}_f_{}.png'.format(idx, f_0)

            fig, axes = plt.subplots(3, 1, figsize=(20, 10))
            plt.suptitle(step5_title, fontsize=20)

            axes[0].plot(ecg_signals[test_sub_idx][idx], 'r', label='ECG')
            axes[0].legend()
            axes[1].plot(swt_step5_error[idx], 'g--', alpha=0.4, label='err')
            axes[1].plot(swt_step5_output[idx], 'b', label='output')
            axes[1].plot(ref_noise[idx], 'r', label='ref')
            axes[1].legend()
            axes[2].plot(swt_step5_filtered[idx], 'r', label='diff')
            axes[2].plot(imu_signals[test_sub_idx][idx], 'b', alpha=0.3, label='BCG')
            axes[2].legend()

            plt.savefig(step5_plot_dir)
            plt.close(fig)

    ## step 6, feature extraction of peaks and throughs
    swt_step6_features = swt_peak_through(swt_step5_filtered, rpeaks[test_sub_idx])
    breakpoint()

if __name__ == '__main__':
    main()