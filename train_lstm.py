import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError

def train_lstm(X_train, y_train, epochs=50, batch_size=32):
    """
    Train an LSTM model to predict states.
    
    Args:
        X_train (np.array): Input sequences for training.
        y_train (np.array): Target states for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        model: Trained LSTM model.
    """
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dense(y_train.shape[1]))  # Output size matches state size

    # Compile the model
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return model

if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")

    # Train the LSTM model
    model = train_lstm(X_train, y_train)

    # Save the trained model
    model.save("state_predictor_model.h5")
    
    print("Model training complete! Model saved as 'state_predictor_model.h5'.")
