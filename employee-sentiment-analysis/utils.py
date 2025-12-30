# utils.py
# Helper functions for the Employee Sentiment Analysis project

import pandas as pd


def load_data(file_path: str = "data/test.xlsx") -> pd.DataFrame:
    """
    Load the Enron email dataset from Excel file.
    Returns a cleaned DataFrame with proper column names and correct dates.
    """
    print("Loading data from:", file_path)

    # Read the Excel file (sheet name is "in")
    df = pd.read_excel(file_path, sheet_name="in")

    print(f"Raw data loaded: {len(df)} rows")
    print("Original columns:", list(df.columns))

    # Rename columns to lowercase
    df.columns = ['subject', 'body', 'date', 'from']

    # Remove rows where body or sender is missing
    df = df.dropna(subset=['body', 'from']).copy()

    print(f"After dropping missing values: {len(df)} rows")

    # The date column is already datetime — just make sure and extract date only
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.date

    # Remove invalid dates (should be none)
    df = df.dropna(subset=['date'])

    # Clean employee email
    df['employee_email'] = df['from'].str.strip().str.lower()

    # Final summary
    print(f"\nSUCCESS: Loaded {len(df)} valid messages")
    print(f"From {df['employee_email'].nunique()} unique employees")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def save_processed_data(df: pd.DataFrame, filename: str = "processed/labeled_data.csv"):
    """
    Save the processed DataFrame to CSV for later use.
    """
    df.to_csv(filename, index=False)
    print(f"Saved processed data to {filename}")