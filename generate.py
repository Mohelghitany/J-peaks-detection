import pandas as pd
import math
import numpy as np
# Define the path to your raw BCG file

import pandas as pd
import os
input_path = r"01_20231105_BCG.csv"
output_path = r"01_20231105_BCG_time.csv"


# Read and process data
with open(input_path, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

# Extract metadata from first data line
header = lines[0].split(',')
first_data_line = lines[1].split(',')
try:
    fs = int(first_data_line[2]) if len(first_data_line) > 2 else 140
    start_timestamp = int(first_data_line[1]) / 1000  # Convert ms to seconds
except (IndexError, ValueError) as e:
    raise ValueError("Error parsing metadata from first data line") from e

# Process BCG values
bcg_values = []
for line in lines[1:]:
    parts = line.split(',')
    if parts and parts[0].lstrip('-').isdigit():
        bcg_values.append(int(parts[0]))

# Generate time vector and epochs
n_samples = len(bcg_values)
time_vector = [start_timestamp + i/fs for i in range(n_samples)]
epochs = [i//140 for i in range(n_samples)]  # 0-based epochs

# Create DataFrame
df = pd.DataFrame({
    'Time (s)': time_vector,
    'BCG Amplitude': bcg_values,
    'epoch': epochs
})

# Save results
df.to_csv(output_path, index=False)
print(f"Successfully processed {n_samples} samples. Output saved to:\n{output_path}")