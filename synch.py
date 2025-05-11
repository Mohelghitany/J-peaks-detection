
from datetime import datetime, timedelta
import math

ecg_data = []
with open('01_20231105_RR.csv', 'r') as f:
    next(f)  # Skip header: "Timestamp,Heart Rate,RR Interval in seconds"
    for line in f:
        parts = line.strip().split(',')
        timestamp_str = parts[0]  # e.g., "2023/11/4 19:12:20"
        heart_rate = int(parts[1])
        rr_interval = float(parts[2])
        
        # Parse local time and convert to UTC (local time is UTC-2, so UTC = local + 2 hours)
        dt_local = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S")
        dt_utc = dt_local + timedelta(hours=2)
        unix_time = dt_utc.timestamp()
        second = int(unix_time)  # Floor to nearest second
        ecg_data.append((second, heart_rate, rr_interval, timestamp_str))

# Remove duplicates in ECG: keep first entry per second
ecg_dict = {}
for entry in ecg_data:
    second = entry[0]
    if second not in ecg_dict:
        ecg_dict[second] = entry
ecg_seconds = set(ecg_dict.keys())  # Set of unique seconds with ECG data
kept_ecg = list(ecg_dict.values())  # List of kept ECG entries

# Load and process BCG data
bcg_data = []
with open('resampled_01_20231105_BCG_time_50Hz.csv', 'r') as f:
    next(f)  # Skip header: "Timestamp,amplitude"
    for line in f:
        parts = line.strip().split(',')
        timestamp = float(parts[0])  # e.g., 1699096655.239 (seconds with ms)
        amplitude = float(parts[1])
        second = int(timestamp)  # Floor to nearest second
        bcg_data.append((timestamp, amplitude, second))

# Filter BCG to keep only samples where second is in ecg_seconds
filtered_bcg = [entry for entry in bcg_data if entry[2] in ecg_seconds]

# Ensure len(ecg) == len(bcg)/50
len_ecg = len(kept_ecg)
len_bcg = len(filtered_bcg)
target_len_ecg = len_bcg // 50  # Number of seconds BCG can support

if len_ecg > target_len_ecg:
    # Trim ECG to match BCG length
    kept_ecg = kept_ecg[:target_len_ecg]
    ecg_seconds = set(entry[0] for entry in kept_ecg)
    filtered_bcg = [entry for entry in filtered_bcg if entry[2] in ecg_seconds]
elif len_ecg < target_len_ecg:
    # Trim BCG to match ECG length
    target_len_bcg = len_ecg * 50
    filtered_bcg = filtered_bcg[:target_len_bcg]
    # Update ecg_seconds if necessary, but should already match
    bcg_seconds = set(entry[2] for entry in filtered_bcg)
    kept_ecg = [entry for entry in kept_ecg if entry[0] in bcg_seconds]

# Verify synchronization
len_ecg = len(kept_ecg)
len_bcg = len(filtered_bcg)
assert len_ecg == math.ceil(len_bcg / 50), f"Lengths mismatch: len(ecg)={len_ecg}, len(bcg)/50={len_bcg/50}"

# Save synchronized ECG data with original column types
with open('ecg_synced.csv', 'w') as f:
    f.write("Timestamp,Heart Rate,RR Interval in seconds\n")
    for entry in kept_ecg:
        timestamp_str = entry[3]  # Original string format
        heart_rate = entry[1]
        rr_interval = entry[2]
        f.write(f"{timestamp_str},{heart_rate},{rr_interval}\n")

# Save synchronized BCG data with original column types
with open('bcg_synced.csv', 'w') as f:
    f.write("Timestamp,amplitude\n")
    for entry in filtered_bcg:
        timestamp = entry[0]  # Original float format
        amplitude = entry[1]
        f.write(f"{timestamp},{amplitude}\n")