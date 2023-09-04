# Import necessary modules and functions
from data_loader import load_eeg_data, preprocess_data
from data_splitter import split_data
from feature_engineering import compute_mean_features, compute_power_features
from model import EEGSeizurePredictionModel

def train_eeg_seizure_prediction_model():
    # Load and preprocess EEG data
    csv_file_path = 'Epileptic Seizure Recognition.csv'  # Replace with the path to your EEG data CSV file
    X, y = load_eeg_data(csv_file_path)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_train, y_train)

    # Perform feature engineering (modify as needed)
    X_train_mean_features = compute_mean_features(X_train)
    X_val_mean_features = compute_mean_features(X_val)
    X_test_mean_features = compute_mean_features(X_test)

    # Initialize the model
    seizure_model = EEGSeizurePredictionModel()

    # Train the model on the training data
    seizure_model.train(X_train_mean_features, y_train)

    # Make predictions on the validation set
    y_val_pred = seizure_model.predict(X_val_mean_features)

    # Evaluate the model's performance on the validation set
    val_accuracy = seizure_model.evaluate(X_val_mean_features, y_val)
    print(f'Validation Accuracy: {val_accuracy:.2f}')

    # Make predictions on the test set
    y_test_pred = seizure_model.predict(X_test_mean_features)

    # Evaluate the model's performance on the test set
    test_accuracy = seizure_model.evaluate(X_test_mean_features, y_test)
    print(f'Test Accuracy: {test_accuracy:.2f}')

if __name__ == "__main__":
    train_eeg_seizure_prediction_model()
