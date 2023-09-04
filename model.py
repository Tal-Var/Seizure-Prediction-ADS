from sklearn.ensemble import RandomForestClassifier

class EEGSeizurePredictionModel:
    def __init__(self, n_estimators=100, random_state=None):
        """
        Initialize the EEG seizure prediction model.

        Args:
            n_estimators (int): Number of decision trees in the random forest.
            random_state (int): Seed for random number generation for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        """
        Train the EEG seizure prediction model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            y_pred (np.array): Predicted labels.
        """
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance on the test set.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels.

        Returns:
            accuracy (float): Accuracy of the model on the test set.
        """
        accuracy = self.model.score(X_test, y_test)
        return accuracy

if __name__ == "__main__":
    # Example usage:
    from data_loader import load_eeg_data, preprocess_data
    from data_splitter import split_data

    # Load and preprocess EEG data (assuming you've already done this)
    csv_file_path = 'Epileptic Seizure Recognition.csv'  # Replace with the path to your EEG data CSV file
    X, y = load_eeg_data(csv_file_path)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_train, y_train)

    # Initialize the model
    seizure_model = EEGSeizurePredictionModel()

    # Train the model on the training data
    seizure_model.train(X_train, y_train)

    # Make predictions on the test set
    y_pred = seizure_model.predict(X_test)

    # Evaluate the model's performance
    accuracy = seizure_model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy:.2f}')
