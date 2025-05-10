"""
Created on %(27/10/2016)
Function to detect bed patterns
"""

import math
import matplotlib
matplotlib.use('agg')
from matplotlib.pyplot import plot, savefig, figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os

# The segmentation is performed based on the standard deviation of each time window
# In general if the std is less than 15, it means tha the there is no any pressure applied to the mat.
# if the std if above 2 * MAD all time-windows SD it means, we are facing body movements.
# On the other hand, if the std is between 15 and 2 * MAD of all time-windows SD,
# there will be a uniform pressure to the mat. Then, we can analyze the sleep patterns

def detect_patterns(pt1, pt2, win_size, data, time, ecg_samples_per_window, plot):
    # Create results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Store initial window points
    pt1_, pt2_ = pt1, pt2

    # Number of windows in BCG data
    limit = int(math.floor(data.size / win_size))
    flag = np.zeros(data.size)  # BCG flag array
    event_flags = np.zeros(limit)  # Window classification

    segments_sd = []

    # Calculate standard deviation per window
    pt1, pt2 = pt1_, pt2_
    for i in range(limit):
        sub_data = data[pt1:pt2]
        segments_sd.append(np.std(sub_data, ddof=1))
        pt1 = pt2
        pt2 += win_size

    # Compute Median Absolute Deviation (MAD)
    mad = np.sum(np.abs(segments_sd - np.mean(segments_sd))) / len(segments_sd)
    thresh1, thresh2 = 15, 2 * mad

    # Classify windows
    pt1, pt2 = pt1_, pt2_
    for j in range(limit):
        std_fos = np.around(segments_sd[j])
        if std_fos < thresh1:  # No-movement
            flag[pt1:pt2] = 3
            event_flags[j] = 3
        elif std_fos > thresh2:  # Movement
            flag[pt1:pt2] = 2
            event_flags[j] = 2
        else:  # Sleeping
            flag[pt1:pt2] = 1
            event_flags[j] = 1
        pt1 = pt2
        pt2 += win_size

    # BCG mask: True for sleeping periods
    bcg_mask = (flag == 1)

    # Expected ECG size based on BCG length
    expected_ecg_size = math.ceil(data.size / 50)
    ecg_mask = np.zeros(expected_ecg_size, dtype=bool)

    # Map BCG windows to ECG samples
    for j in range(limit):
        if event_flags[j] == 1:  # Sleeping
            ecg_start = j * ecg_samples_per_window
            ecg_end = ecg_start + ecg_samples_per_window
            # Ensure we don't exceed ECG array bounds
            ecg_end = min(ecg_end, expected_ecg_size)
            ecg_mask[ecg_start:ecg_end] = True

    filtered_data = data[bcg_mask]
    filtered_time = time[bcg_mask]

    # Plotting (if enabled)
    if plot == 1:
        data_for_plot = data
        width = np.min(data_for_plot)
        height = np.max(data_for_plot) - width if width < 0 else np.max(data_for_plot)
        
        plt.figure(figsize=(12, 6))
        current_axis = plt.gca()
        plt.plot(np.arange(0, data.size)/50, data_for_plot, '-k', linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [mV]')

        pt1, pt2 = pt1_, pt2_
        for j in range(limit):
            sub_time = np.arange(pt1, pt2)/50
            if event_flags[j] == 3:  # No-movement
                current_axis.add_patch(
                    Rectangle((pt1/50, width), win_size/50, height, facecolor="#FAF0BE", alpha=0.2))
            elif event_flags[j] == 2:  # Movement
                current_axis.add_patch(
                    Rectangle((pt1/50, width), win_size/50, height, facecolor="#FF004F", alpha=1.0))
            else:  # Sleeping
                current_axis.add_patch(
                    Rectangle((pt1/50, width), win_size/50, height, facecolor="#00FFFF", alpha=0.2))
            pt1 = pt2
            pt2 += win_size

        plt.grid(True)
        plt.title('Raw BCG Data with Patterns')
        plt.savefig(os.path.join(results_dir, 'rawData.png'))
        plt.close()

    return filtered_data, filtered_time, bcg_mask, ecg_mask