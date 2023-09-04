# Import necessary modules and functions
from data_loader import load_eeg_data, preprocess_data
from data_splitter import split_data
from feature_engineering import compute_mean_features
from model import EEGSeizurePredictionModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_eeg_seizure_model():
    # Load and preprocess EEG data
    csv_file_path = 'eeg_data.csv'  # Replace with the path to your EEG data CSV file
    X, y = load_eeg_data(csv_file_path)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_train, y_train)

    # Perform feature engineering (modify as needed)
    X_train_mean_features = compute_mean_features(X_train)
    X_test_mean_features = compute_mean_features(X_test)

    # Initialize the model
    seizure_model = EEGSeizurePredictionModel()

    # Train the model on the training data
    seizure_model.train(X_train_mean_features, y_train)

    # Make predictions on the test set
    y_test_pred = seizure_model.predict(X_test_mean_features)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Generate a classification report
    report = classification_report(y_test, y_test_pred)
    print(report)

    # Generate a confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print('Confusion Matrix:')
    print(cm)

if __name__ == "__main__":
    evaluate_eeg_seizure_model()
