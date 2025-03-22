import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(csv_file, gap=5, sequence_length=9, test_size=0.2):
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Convert 'states' column from string to list
    data['states'] = data['states'].apply(eval)

    # Prepare input-output pairs
    X = []
    y = []

    for i in range(len(data) - gap * sequence_length):
        input_sequence = [data['states'].iloc[i + j * gap] for j in range(sequence_length)]
        target_state = data['states'].iloc[i + gap * sequence_length]

        X.append(input_sequence)
        y.append(target_state)

    X = np.array(X)
    y = np.array(y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Save training and testing data
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

if __name__ == "__main__":
    preprocess_data("states_episode_0.csv")
