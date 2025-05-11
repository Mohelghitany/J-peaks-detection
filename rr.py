import pandas as pd
from datetime import datetime

# Replace with your actual filename

output_file = "ecg_synced_data_with_epoch.csv"

# Load the CSV file
df = pd.read_csv(r'ecg_synced.csv')

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M:%S')

# Add new 'Epoch' column
df['Epoch'] = df['Timestamp'].apply(lambda x: int(x.timestamp()))

# Save to a new CSV if needed
df.to_csv('output_with_epoch.csv', index=False)
df.to_csv(output_file, index=False)

print("Saved with epoch column:", output_file)
