import numpy as np
import pandas as pd
import pywt
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import argparse
import os

def extract_differentials(window):
    """
    Extract differentials from the window.
    """
    return np.diff(window)

def extract_fourier_transform(window):
    """
    Extract Fourier Transform from the window.
    """
    return np.abs(fft(window))[:len(window) // 2]

def extract_wavelet_transform(window):
    """
    Extract Wavelet Transform features from the window.
    """
    wavelet_coeffs = pywt.wavedec(window, 'db1', level=2)
    wavelet_features = np.concatenate([np.mean(wavelet_coeffs, axis=1), np.var(wavelet_coeffs, axis=1)])
    return wavelet_features

def extract_features_from_window(window, features_to_extract):
    """
    Extract features from a single window of data (30 points over 5 days).
    
    Args:
    - window (np.array): A window of stock prices (30 hourly points).
    - features_to_extract (list): List of feature types to extract (e.g., ['differential', 'fourier', 'wavelet']).

    Returns:
    - features (np.array): A combined feature vector based on the selected features.
    """
    feature_list = []
    
    if 'differential' in features_to_extract:
        feature_list.append(extract_differentials(window))
    
    if 'fourier' in features_to_extract:
        feature_list.append(extract_fourier_transform(window))
    
    if 'wavelet' in features_to_extract:
        feature_list.append(extract_wavelet_transform(window))
    
    # Combine all features into a single feature vector
    if feature_list:
        combined_features = np.concatenate(feature_list)
    else:
        raise ValueError("No valid features selected for extraction.")
    
    return combined_features

def preprocess_data(df, window_size=30, slide_step=6, features_to_extract=['differential', 'fourier', 'wavelet'], save_raw_data=False):
    """
    Preprocess the data using a sliding window to extract features and optionally collect raw data.

    Args:
    - df (pd.DataFrame): DataFrame containing stock data.
    - window_size (int): Number of points per window (30 points = 5 days of data).
    - slide_step (int): Step size for sliding window (default 6 points = 1 day).
    - features_to_extract (list): List of features to extract (e.g., ['differential', 'fourier', 'wavelet']).
    - save_raw_data (bool): Whether to collect and save raw data windows (default: False).
    
    Returns:
    - processed_data (np.array): Processed data with features extracted in sliding windows.
    - raw_data_windows (np.array): Raw data windows corresponding to the extracted features (if save_raw_data is True).
    """
    features_list = []
    raw_data_windows = [] if save_raw_data else None
    scaler = StandardScaler()

    for i in range(0, len(df) - window_size + 1, slide_step):
        window = df['Close'].iloc[i:i + window_size].values
        
        # Ensure the window has the required length
        if len(window) == window_size:
            features = extract_features_from_window(window, features_to_extract)
            features_list.append(features)
            
            # Optionally collect raw data
            if save_raw_data:
                raw_data_windows.append(window)
    
    # Convert lists to NumPy arrays
    processed_data = np.array(features_list)
    scaled_data = scaler.fit_transform(processed_data)

    # Return the processed data and raw data (if needed)
    if save_raw_data:
        raw_data_windows = np.array(raw_data_windows)
        return scaled_data, raw_data_windows
    else:
        return scaled_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess stock data with feature extraction and optional raw data saving.')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file with stock data')
    parser.add_argument('--output_features_csv', type=str, required=True, help='Output CSV file to save processed features')
    parser.add_argument('--output_raw_data_csv', type=str, help='Output CSV file to save raw data windows (optional, only if saving raw data)')
    parser.add_argument('--window_size', type=int, default=30, help='Window size in data points (default: 30 points = 5 days)')
    parser.add_argument('--slide_step', type=int, default=6, help='Slide step in data points (default: 6 points = 1 day)')
    parser.add_argument('--features', type=str, nargs='+', default=['differential', 'fourier', 'wavelet'],
                        choices=['differential', 'fourier', 'wavelet'],
                        help='Features to extract: differential, fourier, wavelet')
    parser.add_argument('--save_raw_data', action='store_true', help='Flag to indicate whether to save raw data windows')

    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(args.input_csv, index_col=0, parse_dates=True)
    
    # Preprocess the data (both features and optionally raw data)
    if args.save_raw_data:
        processed_data, raw_data_windows = preprocess_data(df, window_size=args.window_size, slide_step=args.slide_step, features_to_extract=args.features, save_raw_data=args.save_raw_data)
        
        # Save processed features and raw data windows to CSV files
        np.savetxt(args.output_features_csv, processed_data, delimiter=",")
        if args.output_raw_data_csv:
            np.savetxt(args.output_raw_data_csv, raw_data_windows, delimiter=",")
            print(f"Raw data windows saved to {args.output_raw_data_csv}")
    else:
        processed_data = preprocess_data(df, window_size=args.window_size, slide_step=args.slide_step, features_to_extract=args.features)
        
        # Save only processed features
        np.savetxt(args.output_features_csv, processed_data, delimiter=",")

    print(f"Processed features saved to {args.output_features_csv}")

if __name__ == "__main__":
    main()
