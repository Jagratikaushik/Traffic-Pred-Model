import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

def predict_all_episodes_and_save(model_path, folder_path, output_folder, gap=5, sequence_length=9):
    """
    Predict all testing samples across all episodes, calculate total MSE, and save actual and predicted states to CSV.
    
    Args:
        model_path (str): Path to the trained LSTM model file.
        folder_path (str): Path to the folder containing preprocessed test datasets.
        output_folder (str): Path to save the new CSV files with actual and predicted states.
        gap (int): Gap between timestamps.
        sequence_length (int): Number of input states in a sequence.
    """
    # Load the trained LSTM model
    model = load_model(model_path)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all test dataset files in the folder
    test_files = [f for f in os.listdir(folder_path) if f.endswith("_X_test.npy")]
    
    print(f"Found {len(test_files)} test datasets: {test_files}")

    total_binary_predicted_states = []
    total_binary_actual_states = []

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

        # Create a DataFrame to store actual and predicted states
        episode_name = os.path.splitext(test_file)[0].replace("_X_test", "")
        actual_predicted_states_df = pd.DataFrame(columns=["timestamp", "actual_state", "predicted_state"])

        binary_predicted_states = []
        
        # Predict states for all test samples in this episode
        for i in range(len(X_test)):
            # Generate timestamps for actual states based on gap
            timestamps_actual = [j * gap + 1 + i * gap * (sequence_length + 1) for j in range(sequence_length)]
            timestamp_predicted = timestamps_actual[-1] + gap  # Predicted timestamp (e.g., 46)

            # Add actual states to DataFrame
            for j, timestamp in enumerate(timestamps_actual):
                actual_state = X_test[i][j]
                actual_predicted_states_df = pd.concat(
                    [actual_predicted_states_df, pd.DataFrame({
                        "timestamp": [timestamp],
                        "actual_state": [list(actual_state)],
                        "predicted_state": [None]  # No predictions for actual timestamps
                    })],
                    ignore_index=True,
                )

            # Predict the next state using LSTM model
            input_sequence = X_test[i].reshape(1, X_test.shape[1], X_test.shape[2])  # Reshape for LSTM input
            predicted_state = model.predict(input_sequence)

            # Convert predictions to binary values (0 or 1)
            binary_predicted_state = (predicted_state >= 0.5).astype(int).flatten()

            # Add the predicted state to DataFrame
            actual_predicted_states_df = pd.concat(
                [actual_predicted_states_df, pd.DataFrame({
                    "timestamp": [timestamp_predicted],
                    "actual_state": [None],  # No actual state for predicted timestamp
                    "predicted_state": [list(binary_predicted_state)]
                })],
                ignore_index=True,
            )

            binary_predicted_states.append(binary_predicted_state)

        binary_predicted_states = np.array(binary_predicted_states)

        # Convert actual states to binary values (0 or 1)
        binary_actual_states = (y_test >= 0.5).astype(int)

        # Append binary predictions and actual states to total lists
        total_binary_predicted_states.extend(binary_predicted_states)
        total_binary_actual_states.extend(binary_actual_states)

        # Save the DataFrame as a new CSV file in the output folder
        output_file = os.path.join(output_folder, f"actual_predicted_{episode_name}.csv")
        actual_predicted_states_df.to_csv(output_file, index=False)

        print(f"Processed {test_file}: Saved actual and predicted states to {output_file}")

    # Convert lists to NumPy arrays for MSE calculation
    total_binary_predicted_states = np.array(total_binary_predicted_states)
    total_binary_actual_states = np.array(total_binary_actual_states)

    # Calculate Mean Squared Error (MSE) across all episodes using binary values
    mse = mean_squared_error(total_binary_actual_states.flatten(), total_binary_predicted_states.flatten())
    print("\nTotal Mean Squared Error (MSE) across all episodes:", mse)

if __name__ == "__main__":
    model_path = "state_predictor_model.h5"  # Replace with your trained model path
    folder_path = "."  # Folder containing preprocessed datasets (.npy files)
    output_folder = "csv_actual_predicted"  # Folder to save output files
    
    predict_all_episodes_and_save(model_path, folder_path, output_folder)
