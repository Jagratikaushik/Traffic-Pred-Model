import pandas as pd
import os

def extract_gap_states(input_file, output_file, gap=5):
    """
    Extract rows with timestamps separated by a gap of `gap` from the input CSV file and save to a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
        gap (int): Gap between timestamps.
    """
    # Load the existing CSV file
    data = pd.read_csv(input_file)

    # Ensure the 'timestamp' column is correctly formatted
    data['timestamp'] = data['timestamp'].astype(int)

    # Filter rows where timestamp is spaced by the specified gap
    filtered_data = data[data['timestamp'] % gap == 1]

    # Save the filtered rows to a new CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the output directory exists
    filtered_data.to_csv(output_file, index=False)

    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    input_folder = "csv"  # Folder containing input episode CSV files
    output_folder = "csv_gap_5"  # Folder to save output files with a gap of 5 timestamps

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    # Process each episode file in the input folder
    for episode_file in os.listdir(input_folder):
        if episode_file.endswith(".csv"):
            input_file = os.path.join(input_folder, episode_file)
            output_file = os.path.join(output_folder, f"gap_{episode_file}")
            extract_gap_states(input_file, output_file, gap=5)
