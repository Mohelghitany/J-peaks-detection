import math
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
from ecgdetectors import Detectors

# Main program starts here
print('\nstart processing ...')

file = 'bcg_syncd_Human.csv'
ecg_df = pd.read_csv('ecg_synced_data_with_epoch.csv')

if file.endswith(".csv"):
    fileName = os.path.join(file)
    if os.stat(fileName).st_size != 0:
        bcg_df = pd.read_csv(fileName)
        

        # Extract series
        bcg_data = bcg_df['amplitude'].values
        bcg_time = bcg_df['Timestamp'].values
        ecg_data = ecg_df['Heart Rate'].values  # RR intervals in ms
        ecg_time = ecg_df['Epoch'].values
        start_point, end_point, window_shift, fs = 0, 500, 500, 50  # Adjusted window_shift to 500 for non-overlapping windows
        ecg_samples_per_window = 10
        # Detect and filter patterns
        data_stream, utc_time, bcg_mask, ecg_mask = detect_patterns(
        start_point, end_point, window_shift,
        bcg_data, bcg_time,
        ecg_samples_per_window,
        plot=1
    )

    # Filter ECG data
        filtered_ecg_data = ecg_data[ecg_mask]
        filtered_ecg_time = ecg_time[ecg_mask]
        
        print("Filtered BCG time points:", utc_time[:5])
        print("Filtered ECG time points:", filtered_ecg_time[:5])

        # BCG signal extraction
        movement = band_pass_filtering(data_stream, fs, "bcg")
        # Respiratory signal extraction
        breathing = band_pass_filtering(data_stream, fs, "breath")
        breathing = remove_nonLinear_trend(breathing, 3)
        breathing = savgol_filter(breathing, 11, 3)
        # Wavelet transform for heart rate
        w = modwt(movement, 'bior3.9', 4)
        dc = modwtmra(w, 'bior3.9')
        wavelet_cycle = dc[4]

        # === Heart Rate Estimation ===
        window_length = 500  # 10 seconds at 50 Hz
        window_shift = 500   # Non-overlapping windows
        limit = int(math.floor(len(wavelet_cycle) / window_shift))
        t1, t2, window_length, window_shift = 0, 500, 500, 500
        hop_size = math.floor((window_length - 1) / 2)
        limit = int(math.floor(breathing.size / window_shift))
        # ==========================================================================================================
        # Heart Rate
        beats = vitals(t1, t2, window_shift, limit, wavelet_cycle, utc_time, mpd=1, plot=0)
        print('\nHeart Rate Information')
        print('Minimum pulse : ', np.around(np.min(beats)))
        print('Maximum pulse : ', np.around(np.max(beats)))
        print('Average pulse : ', np.around(np.mean(beats)))
        # Breathing Rate
        beats = vitals(t1, t2, window_shift, limit, breathing, utc_time, mpd=1, plot=0)
        print('\nRespiratory Rate Information')
        print('Minimum breathing : ', np.around(np.min(beats)))
        print('Maximum breathing : ', np.around(np.max(beats)))
        print('Average breathing : ', np.around(np.mean(np.mean(beats))))
        # ==============================================================================================================
        thresh = 0.3
        events = apnea_events(breathing, utc_time, thresh=thresh)
        # ==============================================================================================================
        # Plot Vitals Example
        t1, t2 = 2500, 2500 * 2
        data_subplot(data_stream, movement, breathing, wavelet_cycle, t1, t2)
        ref_hr = filtered_ecg_data
        ref_hr = pd.to_numeric(ref_hr, errors='coerce')
        bcg_hr = np.array(beats, dtype=float)

        min_len = min(len(ref_hr), len(bcg_hr))
        ref_hr = ref_hr[:min_len]
        bcg_hr = bcg_hr[:min_len]

        mask = np.isfinite(ref_hr) & np.isfinite(bcg_hr)
        ref_hr = ref_hr[mask]
        bcg_hr = bcg_hr[mask]

        print("ref_hr (first 10):", ref_hr[:10])
        print("bcg_hr (first 10):", bcg_hr[:10])
        print("ref_hr dtype:", ref_hr.dtype, "min:", np.min(ref_hr), "max:", np.max(ref_hr))
        print("bcg_hr dtype:", bcg_hr.dtype, "min:", np.min(bcg_hr), "max:", np.max(bcg_hr))

       
       

        # Number of BCG HR windows
        n_windows = len(bcg_hr)
        # Number of reference HR samples per window
        samples_per_window = len(ref_hr) // n_windows

        ref_hr_windowed = [
            np.mean(ref_hr[i*samples_per_window:(i+1)*samples_per_window])
            for i in range(n_windows)
        ]
        ref_hr_windowed = np.array(ref_hr_windowed)
        print("ref_hr_windowed (first 10):", ref_hr_windowed[:10])
        print("bcg_hr (first 10):", bcg_hr[:10])
        print("ref_hr_windowed dtype:", ref_hr_windowed.dtype, "min:", np.min(ref_hr_windowed), "max:", np.max(ref_hr_windowed))
        print("bcg_hr dtype:", bcg_hr.dtype, "min:", np.min(bcg_hr), "max:", np.max(bcg_hr))

        # Tune peak detection
        mpd_samples = int(fs * 0.6)  # 0.5 seconds
        window_len = 1000
        local_mph = np.array([
            np.percentile(movement[max(0, i-window_len//2):min(len(movement), i+window_len//2)], 98)
            for i in range(len(movement))
        ])
        j_peak_indices = detect_peaks(
            movement,
            mph=None,
            mpd=mpd_samples,
            threshold=0,
            edge='rising',
            show=False
        )
        j_peak_indices = [i for i in j_peak_indices if movement[i] > local_mph[i]]
        j_peak_indices = np.array(j_peak_indices)

        # --- Tuning parameters ---
        outlier_std = 4        # Outlier threshold (std devs)
        savgol_window = 15      # Smoothing window (must be odd)
        savgol_poly = 4         # Smoothing polynomial order
        hr_min = 40             # Minimum plausible HR (bpm)
        hr_max = 120            # Maximum plausible HR (bpm)
        medfilt_kernel = 11     # Median filter kernel size
        ma_window = 11          # Moving average window size

        # --- Preprocessing ---
        movement_no_outliers = np.clip(
            movement,
            np.mean(movement) - outlier_std * np.std(movement),
            np.mean(movement) + outlier_std * np.std(movement)
        )
        movement_normalized = (movement_no_outliers - np.mean(movement_no_outliers)) / np.std(movement_no_outliers)
        movement_processed = savgol_filter(movement_normalized, savgol_window, savgol_poly)

        # --- Christov detector ---
        detectors = Detectors(fs)
        pt_peaks = detectors.christov_detector(movement_processed)
        pt_peaks = np.array(pt_peaks)
        if len(pt_peaks) > 1:
            pt_intervals = np.diff(pt_peaks) / fs
            pt_hr = 60 / pt_intervals
            pt_hr = pt_hr[(pt_hr > hr_min) & (pt_hr < hr_max)]
        else:
            pt_hr = np.array([])

        # --- HR smoothing ---
        if pt_hr.size > 0:
            pt_hr_smooth = medfilt(pt_hr, kernel_size=medfilt_kernel)
            pt_hr_smooth = np.convolve(pt_hr_smooth, np.ones(ma_window)/ma_window, mode='same')
        else:
            pt_hr_smooth = pt_hr

        plausible_hr_smooth = pt_hr_smooth

        hr_std = [np.std(plausible_hr_smooth[max(0, i-2):i+3]) for i in range(len(plausible_hr_smooth))]
        hr_std = np.array(hr_std)
        good_windows = hr_std < 10  # Only use windows with low HR variability

        min_len = min(len(plausible_hr_smooth), len(ref_hr_windowed), len(good_windows))
        plausible_hr_smooth = plausible_hr_smooth[:min_len]
        ref_hr_windowed = ref_hr_windowed[:min_len]
        good_windows = good_windows[:min_len]

        # Use plausible_hr_smooth for error metrics and plots
        mae = np.mean(np.abs(plausible_hr_smooth[good_windows] - ref_hr_windowed[good_windows]))
        rmse = np.sqrt(np.mean((plausible_hr_smooth[good_windows] - ref_hr_windowed[good_windows]) ** 2))
        mape = np.mean(np.abs((plausible_hr_smooth[good_windows] - ref_hr_windowed[good_windows]) / ref_hr_windowed[good_windows])) * 100

        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # 5. Bland-Altman plot
        mean_hr = (plausible_hr_smooth[good_windows] + ref_hr_windowed[good_windows]) / 2
        diff_hr = plausible_hr_smooth[good_windows] - ref_hr_windowed[good_windows]
        mean_diff = np.mean(diff_hr)
        std_diff = np.std(diff_hr)

        plt.figure(figsize=(8, 5))
        plt.scatter(mean_hr, diff_hr, alpha=0.5)
        plt.axhline(mean_diff, color='red', linestyle='--')
        plt.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle='--')
        plt.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle='--')
        plt.title('Bland-Altman Plot')
        plt.xlabel('Mean HR (bpm)')
        plt.ylabel('Difference (BCG HR - Ref HR)')
        plt.savefig('bland_altman_plot.png')
        plt.close()

        # 6. Pearson correlation plot
        corr, _ = pearsonr(plausible_hr_smooth[good_windows], ref_hr_windowed[good_windows])
        plt.figure(figsize=(8, 5))
        plt.scatter(ref_hr_windowed[good_windows], plausible_hr_smooth[good_windows], alpha=0.5)
        plt.plot([ref_hr_windowed[good_windows].min(), ref_hr_windowed[good_windows].max()], [ref_hr_windowed[good_windows].min(), ref_hr_windowed[good_windows].max()], 'r--')
        plt.title(f'Pearson Correlation: r={corr:.2f}')
        plt.xlabel('Reference HR (bpm)')
        plt.ylabel('Estimated HR (bpm)')
        plt.savefig('correlation_plot.png')
        plt.close()

        # 7. Boxplot
        plt.figure(figsize=(8, 5))
        plt.boxplot([ref_hr_windowed[good_windows], plausible_hr_smooth[good_windows]], tick_labels=['Reference HR', 'Estimated HR'])
        plt.title('Boxplot of HR')
        plt.ylabel('HR (bpm)')
        plt.savefig('your_plot_name.png')
        plt.close()

        plt.figure(figsize=(10,5))
        plt.plot(ref_hr_windowed[good_windows], label='Reference HR')
        plt.plot(plausible_hr_smooth[good_windows], label='BCG HR')
        plt.legend()
        plt.title('HR Comparison')
        plt.xlabel('Window Index')
        plt.ylabel('HR (bpm)')
        plt.savefig('hr_comparison.png')
        plt.close()

        # === LAG DETECTION AND ALIGNMENT ===
        corr = correlate(plausible_hr_smooth - np.mean(plausible_hr_smooth), ref_hr_windowed - np.mean(ref_hr_windowed), mode='full')
        lag = np.argmax(corr) - (len(ref_hr_windowed) - 1)
        print("Best lag (in windows):", lag)
        if lag > 0:
            aligned_bcg_hr = plausible_hr_smooth[lag:]
            aligned_ref_hr = ref_hr_windowed[:-lag]
        elif lag < 0:
            aligned_bcg_hr = plausible_hr_smooth[:lag]
            aligned_ref_hr = ref_hr_windowed[-lag:]
        else:
            aligned_bcg_hr = plausible_hr_smooth
            aligned_ref_hr = ref_hr_windowed
        # Remove NaNs (if any)
        mask = np.isfinite(aligned_bcg_hr) & np.isfinite(aligned_ref_hr)
        aligned_bcg_hr = aligned_bcg_hr[mask]
        aligned_ref_hr = aligned_ref_hr[mask]

        # === FINAL METRICS ===
        mae = np.mean(np.abs(aligned_bcg_hr - aligned_ref_hr))
        rmse = np.sqrt(np.mean((aligned_bcg_hr - aligned_ref_hr) ** 2))
        mape = np.mean(np.abs((aligned_bcg_hr - aligned_ref_hr) / aligned_ref_hr)) * 100
        corr_val, _ = pearsonr(aligned_bcg_hr, aligned_ref_hr)
        print(f"Aligned MAE: {mae:.2f}")
        print(f"Aligned RMSE: {rmse:.2f}")
        print(f"Aligned MAPE: {mape:.2f}%")
        print(f"Aligned Pearson r: {corr_val:.2f}")
        print("Aligned Reference HR std:", np.std(aligned_ref_hr))
        print("Aligned Estimated HR std:", np.std(aligned_bcg_hr))

        # === FINAL PLOTS ===
        # Time series
        plt.figure(figsize=(12, 5))
        plt.plot(aligned_ref_hr, label='Reference HR (aligned)')
        plt.plot(aligned_bcg_hr, label='Estimated BCG HR (aligned)')
        plt.legend()
        plt.title('Aligned HR Time Series')
        plt.xlabel('Window Index')
        plt.ylabel('HR (bpm)')
        plt.savefig('aligned_hr_time_series.png')
        plt.close()

        # Correlation plot
        plt.figure(figsize=(8, 5))
        plt.scatter(aligned_ref_hr, aligned_bcg_hr, alpha=0.5)
        plt.plot([aligned_ref_hr.min(), aligned_ref_hr.max()], [aligned_ref_hr.min(), aligned_ref_hr.max()], 'r--')
        plt.title(f'Aligned Pearson Correlation: r={corr_val:.2f}')
        plt.xlabel('Reference HR (bpm)')
        plt.ylabel('Estimated HR (bpm)')
        plt.savefig('aligned_correlation_plot.png')
        plt.close()

        # Boxplot
        plt.figure(figsize=(8, 5))
        plt.boxplot([aligned_ref_hr, aligned_bcg_hr], tick_labels=['Reference HR', 'Estimated HR'])
        plt.title('Boxplot of HR (Aligned)')
        plt.ylabel('HR (bpm)')
        plt.savefig('aligned_boxplot_hr.png')
        plt.close()

        # Bland-Altman plot
        mean_hr = (aligned_bcg_hr + aligned_ref_hr) / 2
        diff_hr = aligned_bcg_hr - aligned_ref_hr
        mean_diff = np.mean(diff_hr)
        std_diff = np.std(diff_hr)
        plt.figure(figsize=(8, 5))
        plt.scatter(mean_hr, diff_hr, alpha=0.5)
        plt.axhline(mean_diff, color='red', linestyle='--')
        plt.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle='--')
        plt.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle='--')
        plt.title('Bland-Altman Plot (Aligned)')
        plt.xlabel('Mean HR (bpm)')
        plt.ylabel('Difference (BCG HR - Ref HR)')
        plt.savefig('aligned_bland_altman_plot.png')
        plt.close()

        # BCG Segment with Detected J-peaks (unchanged)
        segment_start = 10000  # adjust as needed
        segment_end = segment_start + 2000
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(segment_start, segment_end), movement[segment_start:segment_end], label='Filtered BCG')
        peaks_in_segment = j_peak_indices[(j_peak_indices >= segment_start) & (j_peak_indices < segment_end)]
        plt.plot(peaks_in_segment, movement[peaks_in_segment], 'rx', label='J-peaks')
        plt.legend()
        plt.title('BCG Segment with Detected J-peaks')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.savefig('bcg_segment_with_j_peaks.png')
        plt.close()

    print('\nEnd processing ...')