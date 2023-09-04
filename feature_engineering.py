import numpy as np
import pandas as pd

def toBinary(x):
    if x != 1: return 0;
    else: return 1;
y = y['y'].apply(toBinary)
y = pd.DataFrame(data=y)
y

# Example feature engineering functions (modify as needed)
def compute_mean_features(eeg_data):
    """
    Compute mean features from EEG data.

    Args:
        eeg_data (pd.DataFrame): EEG data with multiple channels.

    Returns:
        mean_features (pd.DataFrame): Mean features for each EEG channel.
    """
    mean_features = eeg_data.mean(axis=0)
    return mean_features

def compute_power_features(eeg_data):
    """
    Compute power spectral density features from EEG data.

    Args:
        eeg_data (pd.DataFrame): EEG data with multiple channels.

    Returns:
        power_features (pd.DataFrame): Power features for each EEG channel.
    """
    # Perform Fourier transform to obtain power spectral density
    fft_result = np.fft.fft(eeg_data, axis=0)
    power_spectral_density = np.abs(fft_result) ** 2

    # Compute power features (e.g., sum of power in specific frequency bands)
    power_features = pd.DataFrame()
    power_features['delta_power'] = np.sum(power_spectral_density[0:4, :], axis=0)
    power_features['theta_power'] = np.sum(power_spectral_density[4:8, :], axis=0)
    power_features['alpha_power'] = np.sum(power_spectral_density[8:13, :], axis=0)
    power_features['beta_power'] = np.sum(power_spectral_density[13:30, :], axis=0)
    power_features['gamma_power'] = np.sum(power_spectral_density[30:, :], axis=0)

    return power_features

if __name__ == "__main__":
    # Example usage:
    from data_loader import load_eeg_data, preprocess_data

    # Load and preprocess EEG data (assuming you've already done this)
    csv_file_path = 'eeg_data.csv'  # Replace with the path to your EEG data CSV file
    X, y = load_eeg_data(csv_file_path)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Example feature extraction
    X_train_mean_features = compute_mean_features(X_train)
    X_train_power_features = compute_power_features(X_train)
