import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_loader import load_eeg_data
from feature_engineering import compute_mean_features
from model import EEGSeizurePredictionModel

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', cmap='Blues'):
    """
    Plot a confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of class labels.
        title (str): Title of the plot.
        cmap (str): Colormap for the plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def visualize_eeg_data(eeg_data):
    """
    Visualize EEG data.

    Args:
        eeg_data (pd.DataFrame): EEG data.
    """
    # Example visualization: Plot EEG data for a single channel
    channel_name = 'channel_1'  # Replace with the name of the channel you want to visualize
    plt.figure(figsize=(12, 4))
    plt.plot(eeg_data[channel_name])
    plt.title(f'EEG Data for {channel_name}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

if __name__ == "__main__":
    # Example usage:

    # Load EEG data
    csv_file_path = 'eeg_data.csv'  # Replace with the path to your EEG data CSV file
    eeg_data, y = load_eeg_data(csv_file_path)

    # Perform feature engineering (modify as needed)
    eeg_data_mean_features = compute_mean_features(eeg_data)

    # Initialize the model
    seizure_model = EEGSeizurePredictionModel()

    # Train the model on the data (if necessary)
    seizure_model.train(eeg_data_mean_features, y)

    # Make predictions (if necessary)
    y_pred = seizure_model.predict(eeg_data_mean_features)

    # Visualize EEG data
    visualize_eeg_data(eeg_data)

    # Example: Visualize a confusion matrix
    plot_confusion_matrix(y, y_pred, labels=['class_0', 'class_1'], title='Confusion Matrix', cmap='Blues')
