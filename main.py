# Import necessary modules and functions
from data_loader import load_eeg_data, preprocess_data
from data_splitter import split_data
from feature_engineering import compute_mean_features
from model import EEGSeizurePredictionModel
from train import train_eeg_seizure_model
from predict import predict_eeg_seizure
from evaluate import evaluate_eeg_seizure_model
from visualizations import plot_confusion_matrix, visualize_eeg_data

def main():
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

    # Make predictions on the test set
    y_test_pred = seizure_model.predict(X_test_mean_features)

    # Evaluate the model's performance
    evaluate_eeg_seizure_model(y_test, y_test_pred)

    # Generate and save visualizations (optional)
    plot_confusion_matrix(y_test, y_test_pred, labels=['class_0', 'class_1'], title='Confusion Matrix', cmap='Blues')
    visualize_eeg_data(X_train)  # Example visualization of EEG data

    # Make predictions on new data (optional)
    new_data_csv_file = 'new_eeg_data.csv'  # Replace with the path to your new EEG data CSV file
    predict_eeg_seizure(new_data_csv_file)

if __name__ == "__main__":
    main()
