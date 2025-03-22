import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

def predict_next_state(model_path, input_sequence):
    """
    Predict the next state using a trained LSTM model.
    
    Args:
        model_path (str): Path to the trained LSTM model file.
        input_sequence (np.array): Input sequence of shape (1, sequence_length, num_features).
    
    Returns:
        predicted_state: Predicted state as a numpy array.
    """
    # Load the trained model
    model = load_model(model_path)

    # Predict the next state
    predicted_state = model.predict(input_sequence)
    
    return predicted_state

if __name__ == "__main__":
    # Load test data
    try:
        X_test = np.load("X_test.npy")  # Input sequences for testing
        y_test = np.load("y_test.npy")  # Actual target states for testing
    except FileNotFoundError:
        print("Test data files 'X_test.npy' or 'y_test.npy' not found! Please run preprocessing first.")
        exit()

    # Initialize lists to store predicted and actual states
    all_predicted_states = []
    all_actual_states = []

    # Loop through all test samples
    for i in range(len(X_test)):
        # Reshape input sequence for prediction
        input_sequence = X_test[i].reshape(1, X_test.shape[1], X_test.shape[2])

        # Predict next state
        predicted_state = predict_next_state("state_predictor_model.h5", input_sequence)

        # Apply threshold to convert predictions to binary values (0 or 1)
        binary_predicted_state = (predicted_state >= 0.5).astype(int)  # Threshold at 0.5

        # Get actual state and convert it to binary values
        actual_state = y_test[i]
        binary_actual_state = (actual_state >= 0.5).astype(int)  # Threshold at 0.5

        # Append predicted and actual states for MSE calculation later
        all_predicted_states.append(predicted_state.flatten())
        all_actual_states.append(actual_state.flatten())

        # Print predicted and actual states
        print(f"Test Sample {i + 1}:")
        print(f"Predicted State (Binary): {binary_predicted_state.flatten()}")
        print(f"Actual State (Binary): {binary_actual_state}")
        print("-" * 50)

    # Convert lists to NumPy arrays for MSE calculation
    all_predicted_states = np.array(all_predicted_states)
    all_actual_states = np.array(all_actual_states)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(all_actual_states, all_predicted_states)
    print("\nMean Squared Error (MSE):", mse)
