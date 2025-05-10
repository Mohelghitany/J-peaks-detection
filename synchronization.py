import math
from datetime import datetime, timedelta
import pandas as pd

def sync_ecg_bcg_dfs(
    ecg_df: pd.DataFrame,
    bcg_df: pd.DataFrame,
    fs: int = 50,
    ecg_tz_offset_hours: int = -2
) -> (pd.DataFrame, pd.DataFrame): # type: ignore
    """
    Synchronize ECG (per-second RR/HR) and BCG (fs Hz) DataFrames.

    Parameters:
    - ecg_df: DataFrame with columns ['Timestamp','Heart Rate','RR Interval in seconds'], Timestamp format "%Y/%m/%d %H:%M:%S" in local time.
    - bcg_df: DataFrame with columns ['Timestamp','amplitude'], Timestamp as float seconds (unix time).
    - fs: sampling frequency of BCG (default 50 Hz).
    - ecg_tz_offset_hours: local timezone offset from UTC for ECG timestamps (default -2 means local = UTC-2).

    Returns:
    - synced_ecg: DataFrame indexed by int-second UTC, length = ceil(len(synced_bcg)/fs).
    - synced_bcg: DataFrame indexed by original BCG timestamp, filtered to only seconds present in ECG and trimmed length.
    """
    # Copy inputs
    ecg = ecg_df.copy()
    bcg = bcg_df.copy()

    # --- Process ECG ---
    # parse local Timestamp, convert to UTC unix seconds
    ecg['dt_local'] = pd.to_datetime(ecg['Timestamp'], format="%Y/%m/%d %H:%M:%S")
    ecg['dt_utc'] = ecg['dt_local'] - pd.to_timedelta(ecg_tz_offset_hours, unit='h')
    ecg['unix'] = ecg['dt_utc'].astype('int64') // 1_000_000_000
    # floor to second and drop duplicates, keep first
    ecg['sec'] = ecg['unix'].astype(int)
    ecg = ecg.drop_duplicates(subset='sec', keep='first')
    ecg = ecg.set_index('sec')

    # --- Process BCG ---
    # floor BCG timestamp to second
    bcg['sec'] = bcg['Timestamp'].astype(float).astype(int)
    # filter to ECG seconds
    bcg = bcg[bcg['sec'].isin(ecg.index)]

    # --- Align lengths ---
    n_ecg = len(ecg)
    n_bcg = len(bcg)
    target_ecg = n_bcg // fs
    if n_ecg > target_ecg:
        # trim ECG
        ecg = ecg.iloc[:target_ecg]
        # re-filter BCG
        bcg = bcg[bcg['sec'].isin(ecg.index)]
    elif n_ecg < target_ecg:
        # trim BCG
        bcg = bcg.iloc[:n_ecg * fs]
        # re-filter ECG
        secs = bcg['sec'].unique()
        ecg = ecg.loc[ecg.index.intersection(secs)]

    # final sanity check
    assert len(ecg) == math.ceil(len(bcg)/fs), \
        f"Lengths mismatch: ECG={len(ecg)}, BCG/fs={len(bcg)/fs}"

    # prepare return DataFrames
    synced_ecg = ecg[['Timestamp','Heart Rate','RR Interval in seconds']].reset_index(drop=True)
    synced_bcg = bcg[['Timestamp','amplitude']].reset_index(drop=True)
    return synced_ecg, synced_bcg

