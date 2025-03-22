import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import os

def predict_all_episodes(model_path, folder_path):
    """
    Predict all testing samples across all episodes and calculate total MSE.
    
    Args:
        model_path (str): Path to the trained LSTM model file.
        folder_path (str): Path to the folder containing preprocessed test datasets.
    """
    # Load the trained LSTM model
    model = load_model(model_path)

    # Get a list of all test dataset files in the folder
    test_files = [f for f in os.listdir(folder_path) if f.endswith("_X_test.npy")]
    
    print(f"Found {len(test_files)} test datasets: {test_files}")

    total_predicted_states = []
    total_actual_states = []

    # Loop through each test dataset file
    for test_file in test_files:
        X_test_file = os.path.join(folder_path, test_file)
        y_test_file = X_test_file.replace("_X_test.npy", "_y_test.npy")

        try:
            # Load test data for this episode
            X_test = np.load(X_test_file)
            y_test = np.load(y_test_file)
        except FileNotFoundError:
            print(f"Test data files not found for {test_file}. Skipping...")
            continue

        print(f"Processing {test_file}...")

        predicted_states = []
        
        # Predict states for all test samples in this episode
        for i in range(len(X_test)):
            input_sequence = X_test[i].reshape(1, X_test.shape[1], X_test.shape[2])  # Reshape for LSTM input
            predicted_state = model.predict(input_sequence)
            predicted_states.append(predicted_state.flatten())

        predicted_states = np.array(predicted_states)

        # Append predictions and actual states to total lists
        total_predicted_states.extend(predicted_states)
        total_actual_states.extend(y_test)

    # Convert lists to NumPy arrays for MSE calculation
    total_predicted_states = np.array(total_predicted_states)
    total_actual_states = np.array(total_actual_states)

    # Calculate Mean Squared Error (MSE) across all episodes
    mse = mean_squared_error(total_actual_states, total_predicted_states)
    print("\nTotal Mean Squared Error (MSE) across all episodes:", mse)

if __name__ == "__main__":
    model_path = "state_predictor_model.h5"  # Replace with your trained model path
    folder_path = "."  # Replace with your folder path containing preprocessed datasets (.npy files)
    
    predict_all_episodes(model_path, folder_path)
