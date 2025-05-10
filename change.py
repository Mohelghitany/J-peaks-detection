import pandas as pd

def change_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with a 'Timestamp' column (seconds since epoch),
    adds a 'human_time' datetime column, and returns the resulting DataFrame.
    """
    # Make a copy so we don’t modify the original in-place
    df_out = df.copy()

    # Convert the 'Timestamp' column to datetime format
    df_out['human_time'] = pd.to_datetime(df_out['Timestamp'], unit='s')

    return df_out

