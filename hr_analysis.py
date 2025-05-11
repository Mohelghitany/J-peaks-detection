import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, find_peaks, savgol_filter, butter, filtfilt, hilbert
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import seaborn as sns
import os

def detect_j_peaks(signal, fs=50, min_distance=0.3):
    """
    Detect J-peaks in BCG signal
    Args:
        signal: BCG signal
        fs: sampling frequency
        min_distance: minimum distance between peaks in seconds
    Returns:
        peak_indices: indices of detected J-peaks
    """
    # Normalize signal
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    # Simple smoothing
    signal_smooth = savgol_filter(signal, 11, 3)
    
    # Find peaks with very lenient parameters
    min_distance_samples = int(min_distance * fs)
    peaks, properties = find_peaks(signal_smooth,
                                 distance=min_distance_samples,
                                 height=0,  # No height threshold
                                 prominence=0.1)  # Very low prominence
    
    # If still no peaks, try even more lenient parameters
    if len(peaks) < 2:
        peaks, properties = find_peaks(signal_smooth,
                                     distance=int(0.2 * fs),  # Shorter minimum distance
                                     height=-0.5,  # Negative threshold
                                     prominence=0.05)  # Extremely low prominence
    
    # Plot for debugging
    plt.figure(figsize=(15, 5))
    plt.plot(signal_smooth, label='Smoothed Signal')
    plt.plot(peaks, signal_smooth[peaks], 'ro', label='Detected Peaks')
    plt.title('Peak Detection')
    plt.grid(True)
    plt.legend()
    
    # Save the debug plot
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'peak_detection_debug.png'))
    plt.close()
    
    return peaks

def calculate_hr(peak_indices, fs=50, window_size=10):
    """
    Calculate heart rate from peak indices
    Args:
        peak_indices: indices of detected peaks
        fs: sampling frequency
        window_size: window size in seconds for HR calculation
    Returns:
        hr: heart rate in bpm
        hr_times: corresponding timestamps
    """
    if len(peak_indices) < 2:
        print("Warning: Not enough peaks detected for HR calculation")
        return np.array([]), np.array([])
    
    # Calculate all RR intervals
    rr_intervals = np.diff(peak_indices) / fs  # in seconds
    
    # Calculate HR from all intervals
    hr = 60 / rr_intervals  # convert to bpm
    
    # Filter out unrealistic HR values (outside 30-200 bpm)
    valid_mask = (hr >= 30) & (hr <= 200)
    hr = hr[valid_mask]
    hr_times = peak_indices[1:][valid_mask] / fs
    
    if len(hr) == 0:
        print("Warning: No valid heart rate values found")
        return np.array([]), np.array([])
    
    return hr, hr_times

def calculate_metrics(hr_estimated, hr_reference):
    """
    Calculate performance metrics
    Args:
        hr_estimated: estimated heart rate values
        hr_reference: reference heart rate values
    Returns:
        metrics: dictionary containing MAE, RMSE, MAPE, and Pearson correlation
    """
    if len(hr_estimated) == 0:
        print("Warning: No heart rate estimates were generated")
        return {
            'MAE': float('inf'),
            'RMSE': float('inf'),
            'MAPE': float('inf'),
            'Pearson_r': 0.0
        }
    
    # Resample reference HR to match estimated HR timestamps
    hr_reference_resampled = np.interp(
        np.arange(len(hr_estimated)), 
        np.linspace(0, len(hr_estimated)-1, len(hr_reference)), 
        hr_reference
    )
    
    mae = mean_absolute_error(hr_reference_resampled, hr_estimated)
    rmse = np.sqrt(mean_squared_error(hr_reference_resampled, hr_estimated))
    mape = np.mean(np.abs((hr_reference_resampled - hr_estimated) / hr_reference_resampled)) * 100
    r, _ = pearsonr(hr_reference_resampled, hr_estimated)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Pearson_r': r
    }

def plot_bland_altman(hr_estimated, hr_reference, save_path):
    """
    Create Bland-Altman plot
    Args:
        hr_estimated: estimated heart rate values
        hr_reference: reference heart rate values
        save_path: path to save the plot
    """
    differences = hr_estimated - hr_reference
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    plt.figure(figsize=(10, 6))
    plt.scatter((hr_estimated + hr_reference)/2, differences, alpha=0.5)
    plt.axhline(y=mean_diff, color='r', linestyle='-')
    plt.axhline(y=mean_diff + 1.96*std_diff, color='g', linestyle='--')
    plt.axhline(y=mean_diff - 1.96*std_diff, color='g', linestyle='--')
    plt.xlabel('Average of Estimated and Reference HR (bpm)')
    plt.ylabel('Difference (Estimated - Reference) (bpm)')
    plt.title('Bland-Altman Plot')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_correlation(hr_estimated, hr_reference, save_path):
    """
    Create correlation plot
    Args:
        hr_estimated: estimated heart rate values
        hr_reference: reference heart rate values
        save_path: path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(hr_reference, hr_estimated, alpha=0.5)
    plt.plot([min(hr_reference), max(hr_reference)], 
             [min(hr_reference), max(hr_reference)], 'r--')
    plt.xlabel('Reference HR (bpm)')
    plt.ylabel('Estimated HR (bpm)')
    plt.title('Correlation Plot')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(hr_estimated, hr_reference, save_path):
    """
    Create boxplot of errors
    Args:
        hr_estimated: estimated heart rate values
        hr_reference: reference heart rate values
        save_path: path to save the plot
    """
    errors = hr_estimated - hr_reference
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=errors)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.ylabel('Error (bpm)')
    plt.title('Distribution of HR Estimation Errors')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def analyze_hr_performance(bcg_signal, reference_hr, fs=50):
    """
    Main function to analyze HR performance
    Args:
        bcg_signal: BCG signal
        reference_hr: reference HR values
        fs: sampling frequency
    """
    # Create results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Detect J-peaks
    peak_indices = detect_j_peaks(bcg_signal, fs)
    
    # Plot detected peaks
    plt.figure(figsize=(12, 6))
    plt.plot(bcg_signal, label='BCG Signal')
    plt.plot(peak_indices, bcg_signal[peak_indices], 'ro', label='Detected Peaks')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('BCG Signal with Detected Peaks')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'detected_peaks.png'))
    plt.close()
    
    # Calculate HR from peaks
    hr_estimated, hr_times = calculate_hr(peak_indices, fs)
    
    if len(hr_estimated) == 0:
        print("\nWarning: No heart rate estimates were generated. This could be due to:")
        print("1. No J-peaks detected in the signal")
        print("2. Signal quality issues")
        print("3. Incorrect sampling frequency")
        return None
    
    # Calculate metrics
    metrics = calculate_metrics(hr_estimated, reference_hr)
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(f"MAE: {metrics['MAE']:.2f} bpm")
    print(f"RMSE: {metrics['RMSE']:.2f} bpm")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"Pearson Correlation: {metrics['Pearson_r']:.3f}")
    
    # Check if metrics meet acceptance criteria
    print("\nAcceptance Criteria Check:")
    print(f"MAE ≤ 3 bpm: {'✓' if metrics['MAE'] <= 3 else '✗'} ({metrics['MAE']:.2f} bpm)")
    print(f"RMSE ≤ 4 bpm: {'✓' if metrics['RMSE'] <= 4 else '✗'} ({metrics['RMSE']:.2f} bpm)")
    print(f"MAPE ≤ 5%: {'✓' if metrics['MAPE'] <= 5 else '✗'} ({metrics['MAPE']:.2f}%)")
    print(f"Pearson r ≥ 0.90: {'✓' if metrics['Pearson_r'] >= 0.90 else '✗'} ({metrics['Pearson_r']:.3f})")
    
    # Create plots
    plot_bland_altman(hr_estimated, reference_hr, os.path.join(results_dir, 'bland_altman.png'))
    plot_correlation(hr_estimated, reference_hr, os.path.join(results_dir, 'correlation.png'))
    plot_error_distribution(hr_estimated, reference_hr, os.path.join(results_dir, 'error_distribution.png'))
    
    return metrics 