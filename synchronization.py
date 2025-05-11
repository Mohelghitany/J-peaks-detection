import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math


def read_ecg(ecg_path: str, utc_offset_hours: float = 2):
    """
    Read ECG CSV and return list of (second, heart_rate, rr_interval, orig_timestamp_str).
    Converts local time (UTC-offset) to UTC and deduplicates by second.
    """
    ecg_data = []
    with open(ecg_path, 'r') as f:
        next(f)
        for line in f:
            ts_str, hr_str, rr_str = line.strip().split(',')
            dt_local = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
            dt_utc = dt_local + timedelta(hours=utc_offset_hours)
            sec = int(dt_utc.timestamp())
            ecg_data.append((sec, int(hr_str), float(rr_str), ts_str))
    # Dedupe: keep first per second
    seen = {}
    for entry in ecg_data:
        if entry[0] not in seen:
            seen[entry[0]] = entry
    return sorted(seen.values(), key=lambda x: x[0])


def read_bcg_raw(bcg_path: str, fs_target: int = 50):
    """
    Read raw BCG CSV, parse initial timestamp and raw fs, then resample to fs_target Hz.
    """
    with open(bcg_path, 'r') as f:
        f.readline()
        hdr = f.readline().strip().split(',')
    ts_offset = int(hdr[1]) / 1000.0
    fs_raw = float(hdr[2])
    df = pd.read_csv(bcg_path, skiprows=2, header=None, names=['amplitude'])

    raw = df['amplitude'].values
    t_raw = ts_offset + np.arange(len(raw)) / fs_raw
    # New uniform grid
    t_new = np.arange(math.ceil(t_raw[0] * fs_target) / fs_target,
                      math.floor(t_raw[-1] * fs_target) / fs_target + 1e-6,
                      1.0/fs_target)
    amp_new = np.interp(t_new, t_raw, raw)
    # Format list
    return [(float(t), float(a), int(t)) for t, a in zip(t_new, amp_new)]


def sync_data(ecg_data, bcg_data, fs: int = 50):
    """
    Align ECG and BCG so that math.ceil(len(bcg)/fs) == len(ecg).
    Trims start/end timestamps at whole-second boundaries.
    """
    # Determine overlapping seconds
    ecg_secs = [e[0] for e in ecg_data]
    bcg_secs = [b[2] for b in bcg_data]
    start = max(min(ecg_secs), min(bcg_secs))
    end = min(max(ecg_secs), max(bcg_secs))

    # Trim by second window
    ecg_trim = [e for e in ecg_data if start <= e[0] <= end]
    bcg_trim = [b for b in bcg_data if start <= b[2] <= end]

    # Compute desired ECG count from trimmed BCG
    n_bcg = len(bcg_trim)
    desired_ecg = math.ceil(n_bcg / fs)
    # Trim ECG to match desired count
    ecg_out = ecg_trim[:desired_ecg]
    # Recompute BCG desired length
    desired_bcg = len(ecg_out) * fs
    bcg_out = bcg_trim[:desired_bcg]

    # Final check
    assert math.ceil(len(bcg_out)/fs) == len(ecg_out), \
        f"Sync failed: ceil({len(bcg_out)}/{fs}) != {len(ecg_out)}"
    return ecg_out, bcg_out


def save_synced(ecg_data, bcg_data, ecg_out: str, bcg_out: str):
    with open(ecg_out, 'w') as f:
        f.write("Timestamp,Heart Rate,RR Interval in seconds\n")
        for _, hr, rr, ts in ecg_data:
            f.write(f"{ts},{hr},{rr}\n")
    with open(bcg_out, 'w') as f:
        f.write("Timestamp,amplitude\n")
        for ts, amp, _ in bcg_data:
            f.write(f"{ts},{amp}\n")


def Resample_sync(
    ecg: pd.DataFrame,
    bcg: pd.DataFrame,
    
) -> (pd.DataFrame, pd.DataFrame): # type: ignore
    

   
    ecg_sync, bcg_sync = sync_data(ecg, bcg, fs=50)
    return ecg_sync, bcg_sync


