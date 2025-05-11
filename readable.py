import pandas as pd

def make_timestamps_human(infile: str, outfile: str, 
                          time_col: str = "timestamp", 
                          fmt: str = "%Y-%m-%d %H:%M:%S.%f") -> None:
    """
    Read infile (with a millisecond UNIX‑epoch column named time_col),
    convert to human‑readable strings, and save to outfile.
    
    Parameters
    ----------
    infile : str
        Path to CSV with columns ['amplitude', time_col].
    outfile : str
        Path where new CSV with ['amplitude', time_col, 'human_time'] is written.
    time_col : str
        Name of the column containing epoch‑ms timestamps.
    fmt : str
        datetime.strftime format for the human_time column.
    """
    # 1) load
    df = pd.read_csv(infile)
    
    # 2) convert ms → datetime
    #    pd.to_datetime will produce a datetime64[ns] column
    df["human_time"] = (
        pd.to_datetime(df[time_col], unit="ms")
          .dt.strftime(fmt)
    )
    
    # 3) save
    df.to_csv(outfile, index=False)
    print(f"Wrote {len(df)} rows with human_time to {outfile}")

# example usage:
make_timestamps_human("bcg_50hz.csv", "bcg_human_time_50hz.csv")