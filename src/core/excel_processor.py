# Assignment 3 
# task 1.1 = Load and Examine Data

import pandas as pd

def load_data():
    """Load and prepare financial data from processed CSV files or raw Excel files."""
    # Load section
    try:
        # Load processed CSV for targets
        df_targets = pd.read_csv('data/processed/KH_Bank_processed.csv')
        print("Loaded from processed CSV successfully.")
        # Select and rename relevant columns
        df_targets = df_targets[['Statement.Entry.Amount.Value', 'Statement.Entry.EntryReference']].copy()
        df_targets.columns = ['Target_Amount', 'Reference_ID']
        print("Renamed columns to Target_Amount and Reference_ID successfully.")
    except FileNotFoundError:
        print("Processed CSV not found. Falling back to raw Excel.")
        df_targets_full = pd.read_excel('data/sample/KH_Bank.XLSX', sheet_name='Sheet1')
        df_targets = df_targets_full[['Statement.Entry.Amount.Value', 'Statement.Entry.EntryReference']].copy()
        df_targets.columns = ['Target_Amount', 'Reference_ID']
        print("Selected and renamed columns from raw Excel successfully.")

    # Load transactions (assuming processed CSV exists)
    try:
        df_transactions = pd.read_csv('data/processed/Customer_Ledger_Entries_FULL_processed.csv')
        print("Loaded transactions from processed CSV successfully.")
    except FileNotFoundError:
        print("Processed transactions CSV not found. Falling back to raw Excel.")
        df_transactions_full = pd.read_excel('data/sample/Customer_Ledger_Entries_FULL.xlsx', sheet_name='Customer Ledger Entries')
        df_transactions = df_transactions_full[['Amount', 'Description']].copy()
        print("Selected transactions columns from raw Excel successfully.")

    # Convert amounts to numeric
    df_transactions['Amount'] = pd.to_numeric(df_transactions['Amount'], errors='coerce')
    df_targets['Target_Amount'] = pd.to_numeric(df_targets['Target_Amount'], errors='coerce')

    # Debug prints
    print("df_transactions columns:", df_transactions.columns.tolist())
    print("df_transactions sample:\n", df_transactions.head())
    print("df_targets columns:", df_targets.columns.tolist())
    print("df_targets sample:\n", df_targets.head())

    # Continue with examination (add your .info(), .describe(), etc. here)
    print("Transactions Info:")
    print(df_transactions.info())
    print(df_transactions.describe())
    print(f"Duplicates in Transactions: {df_transactions.duplicated().sum()}")
    print(f"Unique Descriptions: {df_transactions['Description'].nunique()}")

    print("\nTargets Info:")
    print(df_targets.info())
    print(df_targets.describe())
    print(f"Duplicates in Targets: {df_targets.duplicated().sum()}")
    print(f"Unique Reference IDs: {df_targets['Reference_ID'].nunique()}")

    # Task 1.2: Data Preparation
    print("\nTask 1.2: Data Preparation")

    # Handle missing values
    df_transactions = df_transactions[['Amount', 'Description']].copy()  # Keep only required columns
    df_targets = df_targets[['Target_Amount', 'Reference_ID']].copy()   # Keep only required columns
    print("Dropped unnecessary columns. Checking for missing values...")
    if df_transactions['Amount'].isnull().sum() > 0 or df_transactions['Description'].isnull().sum() > 0:
        df_transactions = df_transactions.dropna(subset=['Amount', 'Description'])
        print(f"Removed {df_transactions['Amount'].isnull().sum() + df_transactions['Description'].isnull().sum()} rows with missing values.")
    else:
        print("No missing values in required columns.")

    if df_targets['Target_Amount'].isnull().sum() > 0 or df_targets['Reference_ID'].isnull().sum() > 0:
        df_targets = df_targets.dropna(subset=['Target_Amount', 'Reference_ID'])
        print(f"Removed {df_targets['Target_Amount'].isnull().sum() + df_targets['Reference_ID'].isnull().sum()} rows with missing values.")
    else:
        print("No missing values in required columns.")

    # Standardize amount formats
    # Convert negatives to positives for consistency with positive targets
    df_transactions['Amount'] = df_transactions['Amount'].abs()
    df_targets['Target_Amount'] = df_targets['Target_Amount'].abs()  # Though already positive
    print("Amounts standardized to positive values.")
    print(f"Transactions with negative amounts: {len(df_transactions[df_transactions['Amount'] < 0])}")
    print(f"Targets with negative amounts: {len(df_targets[df_targets['Target_Amount'] < 0])}")

    # Create unique identifiers
    df_transactions['Transaction_ID'] = range(1, len(df_transactions) + 1)
    df_targets['Target_ID'] = range(1, len(df_targets) + 1)
    print("Added Transaction_ID and Target_ID as unique identifiers.")

    # Save cleaned data
    df_transactions.to_csv('data/processed/clean_transactions.csv', index=False)
    df_targets.to_csv('data/processed/clean_targets.csv', index=False)
    print("Saved cleaned data to 'data/processed/clean_transactions.csv' and 'data/processed/clean_targets.csv'.")

    return df_transactions, df_targets

if __name__ == "__main__":
    # Execute the data loading and preparation when run directly
    load_data()