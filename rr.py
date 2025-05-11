import pandas as pd
from datetime import datetime

def add_epoch_column(df, ts_format: str = '%Y/%m/%d %H:%M:%S') -> pd.DataFrame:
    """
    Takes a DataFrame with a 'Timestamp' column (as strings),
    parses it to datetime using ts_format, adds an 'Epoch' column (int seconds since epoch),
    and returns the resulting DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain a 'Timestamp' column of strings.
    ts_format : str, default '%Y/%m/%d %H:%M:%S'
        The datetime format of the strings in df['Timestamp'].
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame with an added 'Epoch' column.
    """
    df=pd.read_csv(df)
    df_out = df.copy()
    # Parse the Timestamp strings to datetime
    df_out['Timestamp'] = pd.to_datetime(df_out['Timestamp'], format=ts_format)
    # Add the epoch column
    df_out['Epoch'] = df_out['Timestamp'].apply(lambda x: int(x.timestamp()))
    return df_out
