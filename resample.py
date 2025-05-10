import numpy as np
import pandas as pd
from scipy.signal import resample

def resample_bcg_df(
    bcg_df: pd.DataFrame,
    original_fs: int = 140,
    target_fs: int = 50
) -> pd.DataFrame:
    """
    Resample BCG DataFrame from original_fs to target_fs.

    Parameters:
    - bcg_df: DataFrame with columns ['Timestamp', 'amplitude'], Timestamp as float seconds (unix time).
    - original_fs: Original sampling frequency of BCG (default 140 Hz).
    - target_fs: Target sampling frequency of BCG (default 50 Hz).

    Returns:
    - resampled_bcg: DataFrame with resampled 'Timestamp' and 'amplitude' columns.
    """
    # Copy input
    bcg = bcg_df.copy()

    # Extract data
    time = bcg['Timestamp'].values
    amplitude = bcg['amplitude'].values

    # Calculate duration and number of target samples
    duration = time[-1] - time[0]
    n_target = int(duration * target_fs)

    # Resample amplitude
    amplitude_resampled = resample(amplitude, n_target)

    # Create new time vector
    time_resampled = np.linspace(time[0], time[-1], n_target)

    # Create new DataFrame
    resampled_bcg = pd.DataFrame({
        'Timestamp': time_resampled,
        'amplitude': amplitude_resampled
    })

    return resampled_bcg