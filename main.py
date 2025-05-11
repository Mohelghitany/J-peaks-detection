import math
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from ecgdetectors import Detectors
from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_apnea_events import apnea_events
from detect_body_movements import detect_patterns
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot
from detect_peaks import detect_peaks
from scipy.signal import medfilt
from scipy.signal import correlate
from scipy.signal import resample
from sklearn.metrics import mean_absolute_error

def compute_rate(signal, fs, window_sec, shift_sec, min_peak_dist_sec):
    """
    Slide a window through 'signal', detect peaks in each window, 
    and convert to rate in events per minute.

    Parameters
    ----------
    signal : 1-D ndarray
    fs : sampling frequency [Hz]
    window_sec : window length [s]
    shift_sec : window shift [s]
    min_peak_dist_sec : minimum distance between peaks [s]

    Returns
    -------
    rates : ndarray of shape (n_windows,)
        rate in events per minute for each window
    times : ndarray of shape (n_windows,)
        center time [s] of each window
    """
    window_len = int(window_sec * fs)
    shift_len = int(shift_sec * fs)
    mpd = int(min_peak_dist_sec * fs)

    n_windows = (len(signal) - window_len) // shift_len + 1
    rates = np.zeros(n_windows)
    times = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * shift_len
        end = start + window_len
        seg = signal[start:end]

        # detect peaks in this segment
        peaks = detect_peaks(seg, mph=None, mpd=mpd, threshold=0, edge='rising')
        count = len(peaks)

        # convert to per-minute
        rates[i] = count * (60.0 / window_sec)
        # time stamp at center of window
        times[i] = (start + end) / 2.0 / fs

    return times, rates

def compute_ref_hr(ecg_hr, fs, window_sec, shift_sec):
    window_len = int(window_sec * fs)
    shift_len = int(shift_sec * fs)
    n_windows = (len(ecg_hr) - window_len) // shift_len + 1
    ref_rates = np.zeros(n_windows)
    times = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * shift_len
        end = start + window_len
        ref_rates[i] = np.mean(ecg_hr[start:end])
        times[i] = (start + end) / 2.0 / fs
    return times, ref_rates

# Main program starts here
print('\nstart processing ...')

file = 'bcg_syncd_Human.csv'
ecg_df = pd.read_csv('ecg_synced_data_with_epoch.csv')

if file.endswith(".csv"):
    fileName = os.path.join(file)
    if os.stat(fileName).st_size != 0:
        # bcg_df = pd.read_csv(fileName)
        # #---------------------------Generate CSV------------------------------------
        # bcg_df=process_bcg_df(bcg_df)
        # print("GG")
        # print(bcg_df.head())
        # print(ecg_df.head())
        # #---------------------------Resampled------------------------------------
        # bcg_df = resample_bcg_df(bcg_df, original_fs=140, target_fs=50)
        # print("RE")
        # print(bcg_df.head())
        # print(ecg_df.head())

        #  #---------------------------Change Formats-------------------------------
        # ecg_df = add_epoch_column(ecg_df)
        # bcg_df = change_format(bcg_df)
        # print(bcg_df.head())
        # print(ecg_df.head())
        # #---------------------------Sync-----------------------------------------
        # ecg_df,bcg_df=sync_ecg_bcg_dfs(ecg_df,bcg_df)
        # print("sync")
        # print(bcg_df.head())
        # print(ecg_df.head())
        bcg_df = pd.read_csv(fileName)


        bcg_data = bcg_df['amplitude'].values
        bcg_time = bcg_df['Timestamp'].values
        ecg_data = ecg_df['Heart Rate'].values  # RR intervals in ms
        ecg_time = ecg_df['Epoch'].values

        start_point, end_point, window_shift, fs = 0, 500, 500, 50
        ecg_samples_per_window = 10
        # ==========================================================================================================
        data_stream, utc_time, bcg_mask, ecg_mask = detect_patterns(
            start_point, end_point, window_shift,
            bcg_data, bcg_time,
            ecg_samples_per_window,
            plot=1
        )
        # ==========================================================================================================
        # BCG signal extraction
        movement = band_pass_filtering(data_stream, fs, "bcg")
        # ==========================================================================================================
        # Respiratory signal extraction
        breathing = band_pass_filtering(data_stream, fs, "breath")
        breathing = remove_nonLinear_trend(breathing, 3)
        breathing = savgol_filter(breathing, 11, 3)
        # ==========================================================================================================
        w = modwt(movement, 'bior3.9', 4)
        dc = modwtmra(w, 'bior3.9')
        wavelet_cycle = dc[4]
        # ==========================================================================================================
        # Vital Signs estimation - (8 seconds window for better heart rate capture)
        t1, t2, window_length, window_shift = 0, 500, 500, 500
        hop_size = math.floor((window_length - 1) / 2)
        limit = int(math.floor(breathing.size / window_shift))

        hr_times, hr_bcg = compute_rate(
            wavelet_cycle,
            fs=50,              # BCG sampling rate
            window_sec=8,       # 8-second window
            shift_sec=1,        # 1-second shift for overlap
            min_peak_dist_sec=0.4  # enforce ≥400 ms between beats
        )
        print("Heart Rate (bpm):  min {:5.1f}, max {:5.1f}, avg {:5.1f}".format(
            hr_bcg.min(), hr_bcg.max(), hr_bcg.mean()
        ))

        if len(ecg_mask) != len(ecg_data):
            print(f"Adjusting ecg_mask: ecg_data={len(ecg_data)}, ecg_mask={len(ecg_mask)}")
            ecg_mask = ecg_mask[:len(ecg_data)]

        filtered_ecg_data = ecg_data[ecg_mask]
        filtered_ecg_time = ecg_time[ecg_mask]
        ecg_hr_filt = ecg_data[ecg_mask]
        # Filter out invalid heart rates (<= 0)
        valid_mask = ecg_hr_filt > 0
        ecg_hr_filt = ecg_hr_filt[valid_mask]
        filtered_ecg_time = filtered_ecg_time[valid_mask]
        ref_times, hr_ecg = compute_ref_hr(
            ecg_hr_filt, fs=1, window_sec=8, shift_sec=1  # 8-second window, 1-second shift
        )

        print("ECG Heart Rate (bpm):  min {:5.1f}, max {:5.1f}, avg {:5.1f}".format(
            hr_ecg.min(), hr_ecg.max(), hr_ecg.mean()
        ))

        # Apply best shift (1 window) to hr_bcg
        hr_bcg = np.roll(hr_bcg, 1)

        # Align lengths (truncate to shortest)
        n = min(len(hr_bcg), len(hr_ecg))
        hr_bcg = hr_bcg[:n]
        hr_ecg = hr_ecg[:n]

        # Compute metrics
        mae = mean_absolute_error(hr_ecg, hr_bcg)
        corr, pval = pearsonr(hr_ecg, hr_bcg)

        # Print results
        print(f"BCG HR vs ECG HR comparison over {n} windows:")
        print(f"  MAE = {mae:.2f} bpm")
        print(f"  Pearson r = {corr:.3f} (p = {pval:.3g})")
        # ==============================================================================================================
        thresh = 0.3
        events = apnea_events(breathing, utc_time, thresh=thresh)
        # ==============================================================================================================
        # Plot Vitals Example
        t1, t2 = 2500, 2500 * 2
        data_subplot(data_stream, movement, breathing, wavelet_cycle, t1, t2)
        # ==============================================================================================================
    print('\nEnd processing ...')
    # ==================================================================================================================