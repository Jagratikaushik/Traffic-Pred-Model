import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualize_predictions(actual_csv_file, predicted_states):
    """
    Visualize actual vs predicted states.
    """
    actual_data = pd.read_csv(actual_csv_file).iloc[:, 1:].values[:len(predicted_states)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(actual_data[:, 0], label="Actual State")
    plt.plot(predicted_states[:, 0], label="Predicted State", linestyle="--")
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("State Value")
    plt.title("Actual vs Predicted States")
    plt.show()


# Check if both files exist before visualizing
actual_file = "states.csv"
predicted_file = "predicted_states.npy"

if os.path.exists(actual_file) and os.path.exists(predicted_file):
    print(f"Found '{actual_file}' and '{predicted_file}'. Visualizing predictions...")
    
    # Load predicted states and visualize
    predicted_states = np.load(predicted_file)
    visualize_predictions(actual_file, predicted_states)
else:
    raise FileNotFoundError("Both 'states.csv' and 'predicted_states.npy' must exist for visualization.")
