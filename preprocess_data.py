import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_data_for_all_episodes(folder_path, gap=5, sequence_length=9, test_size=0.2):
    """
    Preprocess the data for all episodes to create input-output pairs for LSTM using non-overlapping sequences.
    
    Args:
        folder_path (str): Path to the folder containing CSV files for all episodes.
        gap (int): Gap between timestamps.
        sequence_length (int): Number of input states in a sequence.
        test_size (float): Fraction of data to be used for testing.
    """
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

        # Prepare input-output pairs
        X = []
        y = []

        # Create input-output pairs with non-overlapping sequences
        i = 0
        while i + gap * sequence_length + gap < len(data):
            # Extract input sequence of length `sequence_length`
            input_sequence = [data['states'].iloc[i + j * gap] for j in range(sequence_length)]
            # Extract target state (the state after the last timestamp in the sequence)
            target_state = data['states'].iloc[i + gap * sequence_length]

            X.append(input_sequence)
            y.append(target_state)

            # Move to the next block of timestamps
            i += gap * (sequence_length + 1)

        X = np.array(X)
        y = np.array(y)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Save training and testing data for this episode
        episode_name = os.path.splitext(episode_file)[0]  # Extract file name without extension
        np.save(f"{episode_name}_X_train.npy", X_train)
        np.save(f"{episode_name}_y_train.npy", y_train)
        np.save(f"{episode_name}_X_test.npy", X_test)
        np.save(f"{episode_name}_y_test.npy", y_test)

        print(f"Processed {episode_file}:")
        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

if __name__ == "__main__":
    folder_path = "csv"  # Replace with your actual folder path containing episode CSV files
    preprocess_data_for_all_episodes(folder_path)
