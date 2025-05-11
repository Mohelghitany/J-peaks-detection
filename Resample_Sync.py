import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import os
import matplotlib.pyplot as plt

def read_ecg(ecg_path: str, utc_offset_hours: float = 2):
    """
    Read ECG CSV and return list of (unix_time, heart_rate, rr_interval, orig_timestamp_str).
    Converts local time (UTC-offset) to UTC.
    """
    ecg_data = []
    with open(ecg_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            timestamp_str, heart_rate, rr_interval = parts[0], int(parts[1]), float(parts[2])
            try:
                dt_local = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S")
            except ValueError:
                dt_local = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            dt_utc = dt_local + timedelta(hours=utc_offset_hours)
            unix_time = dt_utc.timestamp()
            ecg_data.append((unix_time, heart_rate, rr_interval, timestamp_str))
    
    # Remove duplicates by unix_time and sort
    ecg_dict = {entry[0]: entry for entry in ecg_data}
    return sorted(ecg_dict.values(), key=lambda x: x[0])

def read_bcg_raw(bcg_path: str, fs_target: int = 50):
    """
    Read raw BCG CSV and resample to target frequency.
    Returns list of (unix_time, amplitude, second) tuples.
    """
    with open(bcg_path, 'r') as f:
        next(f)  # Skip header
        meta = next(f).strip().split(',')
    
    ts_offset_ms, fs_raw = int(meta[1]), float(meta[2])
    ts_offset = ts_offset_ms / 1000.0

    # Read BCG values
    bcg_values = pd.read_csv(bcg_path, skiprows=2, header=None)[0].values
    
    # Original time vector
    t_raw = ts_offset + np.arange(len(bcg_values)) / fs_raw
    
    # Resample to target frequency
    t_new = np.arange(t_raw[0], t_raw[-1], 1.0/fs_target)
    amp_new = np.interp(t_new, t_raw, bcg_values)
    
    return [(t, a, int(t)) for t, a in zip(t_new, amp_new)]

def sync_data(ecg_data, bcg_data, fs: int = 50):
    """
    Synchronize ECG and BCG data with improved alignment.
    Returns synchronized (ecg_data, bcg_data) pair.
    """
    # Find common time window
    ecg_times = [e[0] for e in ecg_data]
    bcg_times = [b[0] for b in bcg_data]
    
    start_time = max(min(ecg_times), min(bcg_times))
    end_time = min(max(ecg_times), max(bcg_times))
    
    if start_time >= end_time:
        raise ValueError("No overlapping time period between BCG and ECG data")
    
    # Filter data to common time window
    ecg_sync = [e for e in ecg_data if start_time <= e[0] <= end_time]
    bcg_sync = [b for b in bcg_data if start_time <= b[0] <= end_time]
    
    # Verify sample ratio
    expected_ratio = fs
    actual_ratio = len(bcg_sync) / len(ecg_sync)
    
    print(f"\nSynchronization Report:")
    print(f"Time window: {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")
    print(f"Duration: {end_time-start_time:.2f} seconds")
    print(f"ECG samples: {len(ecg_sync)}, BCG samples: {len(bcg_sync)}")
    print(f"Expected ratio: {expected_ratio}:1, Actual ratio: {actual_ratio:.2f}:1")
    
    if not (0.95 * expected_ratio <= actual_ratio <= 1.05 * expected_ratio):
        print("Warning: Sample ratio differs significantly from expected")
    
    return ecg_sync, bcg_sync

def save_synced(ecg_data, bcg_data):
    """Save synchronized data to CSV files."""
    
    
    # Save ECG data
    ecg_df = pd.DataFrame([{
        'Timestamp': e[3],
        'HR': e[1],
        'RR': e[2],
        'unix_time': e[0]
    } for e in ecg_data])
    ecg_path = os.path.join('ecg_synced.csv')
    ecg_df.to_csv(ecg_path, index=False)
    
    # Save BCG data
    bcg_df = pd.DataFrame([{
        'timestamp': b[0],
        'amplitude': b[1]
    } for b in bcg_data])
    bcg_path = os.path.join( 'bcg_synced.csv')
    bcg_df.to_csv(bcg_path, index=False)
    
    print(f"\nSaved synchronized data to:")
    print(f"- ECG: {ecg_path}")
    print(f"- BCG: {bcg_path}")

def plot_verification(ecg_data, bcg_data, fs: int = 50, plot_seconds: int = 30):
    """Generate verification plots for the first N seconds."""
    start_time = ecg_data[0][0]
    
    # Prepare ECG data
    ecg_times = [e[0] - start_time for e in ecg_data]
    ecg_hr = [e[1] for e in ecg_data]
    ecg_rr = [e[2] for e in ecg_data]
    
    # Prepare BCG data
    bcg_times = [b[0] - start_time for b in bcg_data]
    bcg_amp = [b[1] for b in bcg_data]
    
    # Select first N seconds
    mask = np.array(bcg_times) <= plot_seconds
    bcg_times_plot = np.array(bcg_times)[mask]
    bcg_amp_plot = np.array(bcg_amp)[mask]
    
    ecg_mask = np.array(ecg_times) <= plot_seconds
    ecg_times_plot = np.array(ecg_times)[ecg_mask]
    ecg_hr_plot = np.array(ecg_hr)[ecg_mask]
    ecg_rr_plot = np.array(ecg_rr)[ecg_mask]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot BCG signal
    plt.subplot(3, 1, 1)
    plt.plot(bcg_times_plot, bcg_amp_plot, 'b-', linewidth=0.5)
    plt.title(f'BCG Signal (First {plot_seconds} Seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot RR intervals
    plt.subplot(3, 1, 2)
    plt.stem(ecg_times_plot, ecg_rr_plot, linefmt='r-', markerfmt='ro', basefmt=" ")
    plt.title('RR Intervals')
    plt.ylabel('RR (seconds)')
    plt.grid(True)
    
    # Plot Heart Rate
    plt.subplot(3, 1, 3)
    plt.plot(ecg_times_plot, ecg_hr_plot, 'ro-', markersize=4)
    plt.title('Heart Rate')
    plt.xlabel('Time (seconds)')
    plt.ylabel('HR (bpm)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def Resample_sync(ecg_path,bcg_path):
    
   
    
    # Read data
    print("Loading ECG data...")
    ecg_data = read_ecg(ecg_path)
    print("Loading and resampling BCG data...")
    bcg_data = read_bcg_raw(bcg_path)
    
    # Synchronize
    print("\nSynchronizing data...")
    ecg_sync, bcg_sync = sync_data(ecg_data, bcg_data)
    
    # Save results
    save_synced(ecg_sync, bcg_sync)
    
    
    return 'ecg_synced.csv', 'bcg_synced.csv'
