#Assignment 2

import pandas as pd
import openpyxl
import numpy as np
import sqlite3
import re
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(
    filename='parsing_errors.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------- Parsing Functions ---------------------

def parse_amount(s):
    """
    Parses financial amounts supporting:
    - US: $1,234.56
    - Europe: €1.234,56
    - India: ₹1,23,456.78
    - Negative parentheses (1,234.56)
    - Trailing negative 1234.56-
    - Abbreviations like 1.2K, 2.5M, 1.2B
    """
    if pd.isna(s):
        return None
    if not isinstance(s, str):
        s = str(s).strip()

    if s == '':
        return None

    try:
        # Handle negative in parentheses
        if s.startswith('(') and s.endswith(')'):
            val = parse_amount(s[1:-1])
            return -val if val is not None else None

        # Handle trailing negative sign
        if s.endswith('-'):
            val = parse_amount(s[:-1])
            return -val if val is not None else None

        # Detect currency symbol and locale format
        currency_symbols = ['$', '€', '₹', '£', '¥']
        currency_symbol = None
        for sym in currency_symbols:
            if sym in s:
                currency_symbol = sym
                s = s.replace(sym, '')
                break

        s = s.strip()

        # Detect abbreviation (K, M, B)
        multiplier = 1
        if len(s) > 0 and s[-1].upper() in ['K', 'M', 'B']:
            suffix = s[-1].upper()
            if suffix == 'K':
                multiplier = 1_000
            elif suffix == 'M':
                multiplier = 1_000_000
            elif suffix == 'B':
                multiplier = 1_000_000_000
            s = s[:-1].strip()

        # Handle different thousand and decimal separators based on currency/locale guess
        # Strategy:
        # - US: ',' thousands, '.' decimal
        # - Europe: '.' thousands, ',' decimal
        # - India: ',' thousands (with groups of 3 and 2), '.' decimal
        # Default to US style if unsure

        if currency_symbol == '€':
            # European format, e.g., 1.234,56
            s = s.replace('.', '').replace(',', '.')
        elif currency_symbol == '₹':
            # Indian format: Remove commas (which are thousand separators)
            s = s.replace(',', '')
        else:
            # Assume US style: remove commas
            s = s.replace(',', '')

        val = float(s)
        return val * multiplier

    except Exception as e:
        logging.warning(f"Amount parse error: '{s}' - {e}")
        return None


def parse_date(s):
    """
    Parses various date formats including:
    - MM/DD/YYYY, DD/MM/YYYY
    - YYYY-MM-DD, DD-MON-YYYY
    - Quarter 1 2024, Q1-24
    - Mar 2024, March 2024
    - Excel serial dates (44927 = Jan 1, 2023)
    - Dec-23 or similar two-digit years
    """
    if pd.isna(s):
        return None
    if isinstance(s, (datetime, pd.Timestamp)):
        return s.date()

    s = str(s).strip()

    # Excel serial date (numbers only)
    if s.isdigit():
        try:
            base_date = datetime(1899, 12, 30)
            date_val = base_date + timedelta(days=int(s))
            return date_val.date()
        except Exception as e:
            logging.warning(f"Excel serial date parse error: '{s}' - {e}")

    # Quarter formats
    quarter_match = re.match(r'(Q|Quarter)[\s\-]?(\d)[\s\-]?(\d{2,4})', s, re.I)
    if quarter_match:
        q = int(quarter_match.group(2))
        year = int(quarter_match.group(3))
        if year < 100:
            year += 2000  # convert 2-digit year to 4-digit
        month = (q - 1) * 3 + 1
        return datetime(year, month, 1).date()

    # Month-Year formats like "Dec-23", "Mar 2024", "March 2024"
    month_year_match = re.match(r'([A-Za-z]+)[\s\-]?(\d{2,4})', s)
    if month_year_match:
        mon_str, yr_str = month_year_match.groups()
        try:
            if len(yr_str) == 2:
                yr_str = '20' + yr_str
            dt = datetime.strptime(f'1 {mon_str} {yr_str}', '%d %b %Y')
            return dt.date()
        except:
            try:
                dt = datetime.strptime(f'1 {mon_str} {yr_str}', '%d %B %Y')
                return dt.date()
            except Exception as e:
                logging.warning(f"Month-Year parse error: '{s}' - {e}")

    # Try known explicit formats
    date_formats = [
        '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%b-%Y',
        '%b %Y', '%B %Y', '%Y%m%d', '%d-%m-%Y', '%d/%m/%y',
        '%m-%d-%Y', '%m-%d-%y'
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(s, fmt).date()
        except:
            continue

    # Fallback parse using pandas
    try:
        parsed = pd.to_datetime(s, errors='coerce')
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception as e:
        logging.warning(f"Date parse error: '{s}' - {e}")
        return None


# Safe wrapper functions for applying to dataframe columns

def safe_parse_amount(val):
    try:
        return parse_amount(val)
    except Exception as e:
        logging.warning(f"Safe parse amount error on value: {val} - {e}")
        return None


def safe_parse_date(val):
    try:
        return parse_date(val)
    except Exception as e:
        logging.warning(f"Safe parse date error on value: {val} - {e}")
        return None


# --------------------- Data Type Detection ---------------------

def detect_column_type(col_data):
    sample = col_data.dropna().astype(str).head(100)

    if len(sample) == 0:
        return 'string', 1.0  # No data = treat as string by default

    date_parsed = sample.apply(parse_date).dropna()
    date_confidence = len(date_parsed) / len(sample) if len(sample) > 0 else 0

    number_parsed = sample.apply(parse_amount).dropna()
    number_confidence = len(number_parsed) / len(sample) if len(sample) > 0 else 0

    if max(date_confidence, number_confidence) < 0.5:
        return 'string', 1.0

    if date_confidence >= number_confidence:
        return 'date', date_confidence
    else:
        return 'number', number_confidence


# --------------------- Advanced Storage ---------------------

class FinancialDataStore:
    def __init__(self):
        self.dataframes = {}       # Store raw dataframes by dataset name
        self.column_types = {}     # Store detected types by dataset name
        self.indexes = {}          # Store indexes (multi-indexed DataFrames) for fast queries

    def add_dataset(self, name, df, column_types):
        self.dataframes[name] = df
        self.column_types[name] = column_types

        # Build MultiIndex for fast querying if date and amount present
        index_cols = []
        if any(col for col, typ in column_types.items() if typ == 'date'):
            date_cols = [col for col, typ in column_types.items() if typ == 'date']
            index_cols += date_cols
        if any(col for col, typ in column_types.items() if typ == 'number'):
            number_cols = [col for col, typ in column_types.items() if typ == 'number']
            # Pick the first number col as amount index
            index_cols += number_cols[:1]

        if index_cols:
            try:
                df_indexed = df.set_index(index_cols)
                self.indexes[name] = df_indexed.sort_index()
            except Exception as e:
                logging.warning(f"Failed to build MultiIndex for {name}: {e}")
                self.indexes[name] = df
        else:
            self.indexes[name] = df

    def query_by_date_range(self, dataset_name, date_col, start_date, end_date):
        if dataset_name not in self.indexes:
            raise ValueError(f"Dataset {dataset_name} not found")

        df = self.indexes[dataset_name]

        if date_col not in df.index.names:
            raise ValueError(f"{date_col} is not an index in dataset {dataset_name}")

        # Slicing by MultiIndex date range
        try:
            # For single-level date index
            if isinstance(df.index, pd.DatetimeIndex) or date_col == df.index.name:
                return df.loc[start_date:end_date]
            # For MultiIndex with date_col included
            return df.loc[(slice(start_date, end_date),), :]
        except Exception as e:
            logging.warning(f"Date range query error: {e}")
            return pd.DataFrame()

    def aggregate(self, dataset_name, group_by_cols, agg_col, agg_func='sum'):
        if dataset_name not in self.dataframes:
            raise ValueError(f"Dataset {dataset_name} not found")

        df = self.dataframes[dataset_name]

        if not all(col in df.columns for col in group_by_cols + [agg_col]):
            raise ValueError("Columns for aggregation not found")

        if agg_func not in ['sum', 'mean', 'count']:
            raise ValueError("Aggregation function not supported")

        grouped = df.groupby(group_by_cols)[agg_col]

        if agg_func == 'sum':
            return grouped.sum()
        elif agg_func == 'mean':
            return grouped.mean()
        elif agg_func == 'count':
            return grouped.count()

    def preview(self, dataset_name, n=5):
        if dataset_name in self.dataframes:
            return self.dataframes[dataset_name].head(n)
        return None


# --------------------- Main ---------------------

def load_excel(file_path):
    xls = pd.ExcelFile(file_path, engine='openpyxl')
    print(f"Loaded file: {file_path} with sheets: {xls.sheet_names}")
    return xls


def process_sheet(xls, sheet_name):
    print(f"Processing {xls.io} - Sheet: {sheet_name}")
    df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
    print(f"Sheet shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def detect_types(df):
    detected_types = {}
    for col in df.columns:
        col_type, confidence = detect_column_type(df[col])
        detected_types[col] = col_type
        print(f"Column '{col}': detected_type={col_type} with confidence={confidence:.2f}")
    return detected_types


def clean_and_parse_df(df, detected_types):
    for col, dtype in detected_types.items():
        if dtype == 'number':
            df[col] = df[col].apply(safe_parse_amount)
        elif dtype == 'date':
            df[col] = df[col].apply(safe_parse_date)
    return df


def main():
    # Your sample files (adjust paths accordingly)
    file_paths = [
        r"N:\financial-data-parser\data\sample\KH_Bank.XLSX",
        r"N:\financial-data-parser\data\sample\Customer_Ledger_Entries_FULL.xlsx"
    ]

    storage = FinancialDataStore()

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        xls = load_excel(file_path)
        for sheet in xls.sheet_names:
            df = process_sheet(xls, sheet)
            detected_types = detect_types(df)
            df_clean = clean_and_parse_df(df, detected_types)

            # Manual override for some columns to string
            force_string_columns = [
                'GroupHeader.MessageIdentification',
                'Statement.Entry.EntryReference',
                'Entry No.'
            ]
            for col in force_string_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str)

            print(f"Sample data after parsing for sheet '{sheet}':")
            print(df_clean.head(3))

            # Add dataset to storage
            storage.add_dataset(sheet, df_clean, detected_types)

            # Example aggregation if possible
            if 'Amount' in df_clean.columns and 'Currency Code' in df_clean.columns:
                try:
                    result = storage.aggregate(sheet, ['Currency Code'], 'Amount', 'sum')
                    print(f"Sum of Amount grouped by Currency Code:\n{result}")
                except Exception as e:
                    print(f"Aggregation failed: {e}")

            # Example date range query on first date column
            date_cols = [c for c, t in detected_types.items() if t == 'date']
            if date_cols:
                date_col = date_cols[0]
                try:
                    # Just demo query last 30 days from max date if available
                    max_date = df_clean[date_col].max()
                    if max_date:
                        start_date = max_date - timedelta(days=30)
                        subset = storage.query_by_date_range(sheet, date_col, start_date, max_date)
                        print(f"Data rows in last 30 days for '{date_col}': {len(subset)}")
                except Exception as e:
                    print(f"Date range query failed: {e}")


# --------------------- Unit Tests ---------------------

def test_parse_amount():
    test_cases = [
        ("$1,234.56", 1234.56),
        ("(2,500.00)", -2500.00),
        ("€1.234,56", 1234.56),
        ("1.5M", 1500000),
        ("₹1,23,456", 123456),
        ("1234.56-", -1234.56),
        ("", None),
        (None, None),
        ("Invalid123", None),
        ("2B", 2000000000)
    ]
    for val, expected in test_cases:
        result = parse_amount(val)
        assert (result == expected or (result is None and expected is None)), f"Fail: {val} => {result}, expected {expected}"

def test_parse_date():
    test_cases = [
        ("12/31/2023", datetime(2023, 12, 31).date()),
        ("2023-12-31", datetime(2023, 12, 31).date()),
        ("Q4 2023", datetime(2023, 10, 1).date()),
        ("Dec-23", datetime(2023, 12, 1).date()),
        ("44927", datetime(2023, 1, 1).date()),  # Excel serial
        ("InvalidDate", None),
        (None, None)
    ]
    for val, expected in test_cases:
        result = parse_date(val)
        assert (result == expected or (result is None and expected is None)), f"Fail: {val} => {result}, expected {expected}"

def run_tests():
    print("Running tests...")
    test_parse_amount()
    test_parse_date()
    print("All tests passed!")


if __name__ == '__main__':
    # Uncomment the line below to run unit tests separately
    # run_tests()

    main()
