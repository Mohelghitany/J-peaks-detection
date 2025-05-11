


# Import required libraries
import math
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, resample

from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_apnea_events import apnea_events
from detect_body_movements import detect_patterns
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot


# Path to input CSV (140 Hz sampled data)
input_path = r"01_20231105_BCG_time.csv"

# Sampling rates
original_fs = 140
target_fs = 50

# Load original CSV file
df = pd.read_csv(input_path)

# Extract data
time = df["Time (s)"].values
amplitude = df["BCG Amplitude"].values

# Duration and number of target samples
duration = time[-1] - time[0]
n_target = int(duration * target_fs)

# Resample amplitude
amplitude_resampled = resample(amplitude, n_target)

# Create new time vector
time_resampled = np.linspace(time[0], time[-1], n_target)

# Create new DataFrame and save to CSV
df_resampled = pd.DataFrame({
    "Timestamp": time_resampled,
    "amplitude": amplitude_resampled
})
df_resampled.to_csv("resampled_01_20231105_BCG_time_50Hz.csv", index=False)

