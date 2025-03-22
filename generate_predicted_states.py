import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def generate_predicted_states(folder_path, output_folder, model_path, gap=5, sequence_length=9):
    """
    Generate predicted states and save them in a new CSV folder.
    
    Args:
        folder_path (str): Path to the folder containing episode CSV files.
        output_folder (str): Path to save the new CSV files with predicted states.
        model_path (str): Path to the trained LSTM model file.
        gap (int): Gap between timestamps.
        sequence_length (int): Number of input states in a sequence.
    """
    # Load the trained LSTM model
    model = load_model(model_path)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of all episode files in the folder
    episode_files = [f for f in os.listdir(folder_path) if f.startswith("states_episode") and f.endswith(".csv")]
    
    print(f"Found {len(episode_files)} episode files: {episode_files}")

    # Loop through each episode file
    for episode_file in episode_files:
        csv_file = os.path.join(folder_path, episode_file)
        
        # Load the CSV file
        data = pd.read_csv(csv_file)

        # Convert 'states' column from string to list
        data['states'] = data['states'].apply(eval)

        # Create a DataFrame to store actual and predicted states
        predicted_states_df = pd.DataFrame(columns=["timestamp", "actual_state", "predicted_state"])

        i = 0
        while i + gap * sequence_length + gap < len(data):
            # Extract input sequence of length `sequence_length`
            input_sequence = [data['states'].iloc[i + j * gap] for j in range(sequence_length)]
            target_timestamp = i + gap * sequence_length + 1  # Timestamp for the predicted state
            
            # Predict next state using LSTM model
            input_sequence_array = np.array(input_sequence).reshape(1, sequence_length, -1)  # Reshape for LSTM input
            predicted_state = model.predict(input_sequence_array).flatten()

            # Save actual and predicted states to DataFrame
            for j in range(sequence_length):
                actual_timestamp = i + j * gap + 1
                actual_state = data['states'].iloc[i + j * gap]
                predicted_states_df = pd.concat(
                    [predicted_states_df, pd.DataFrame({
                        "timestamp": [actual_timestamp],
                        "actual_state": [actual_state],
                        "predicted_state": [None]  # No predictions for actual timestamps
                    })],
                    ignore_index=True,
                )
            
            # Add the predicted state to the DataFrame
            predicted_states_df = pd.concat(
                [predicted_states_df, pd.DataFrame({
                    "timestamp": [target_timestamp],
                    "actual_state": [None],  # No actual state for predicted timestamp
                    "predicted_state": [list(predicted_state)]
                })],
                ignore_index=True,
            )

            # Move to the next block of timestamps
            i += gap * (sequence_length + 1)

        # Save the DataFrame as a new CSV file in the output folder
        output_file = os.path.join(output_folder, f"predicted_{episode_file}")
        predicted_states_df.to_csv(output_file, index=False)

        print(f"Processed {episode_file}: Saved predicted states to {output_file}")

if __name__ == "__main__":
    folder_path = "csv"  # Folder containing input episode CSV files
    output_folder = "csv_predicted"  # Folder to save output files with predicted states
    model_path = "state_predictor_model.h5"  # Path to your trained LSTM model
    
    generate_predicted_states(folder_path, output_folder, model_path)
