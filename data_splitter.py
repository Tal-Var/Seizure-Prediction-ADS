import numpy as np
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
x_test.iloc[1]

def split_data(X, y, test_size=0.2, validation_size=0.2, random_state=42):
    """
    Split EEG data into training, validation, and test sets.

    Args:
        X (pd.DataFrame): EEG features.
        y (pd.Series): Labels.
        test_size (float): Proportion of the dataset to include in the test split.
        validation_size (float): Proportion of the training set to include in the validation split.
        random_state (int): Seed for random number generation.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        y_test (pd.Series): Testing labels.
    """
    # First, split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Next, split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Example usage:
    from data_loader import load_eeg_data, preprocess_data

    # Load and preprocess EEG data (assuming you've already done this)
    csv_file_path = 'Epileptic Seizure Recognition.csv'  # Replace with the path to your EEG data CSV file
    X, y = load_eeg_data(csv_file_path)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_train, y_train)




     
