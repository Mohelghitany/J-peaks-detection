
import pandas as pd
# your original file
output_path = 'bcg_syncd_Human.csv'
df=pd.read_csv('bcg_synced.csv')
df["human_time"] = pd.to_datetime(df["Timestamp"], unit="s")
df.to_csv(output_path, index=False)

