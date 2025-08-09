import re
import pandas as pd
from datetime import datetime, timedelta

class FormatParser:
    @staticmethod
    def parse_amount(val):
        if pd.isna(val):
            return None
        val_str = str(val).strip()

        # Handle negative parentheses
        if re.match(r'^\(.*\)$', val_str):
            val_str = '-' + val_str[1:-1]

        # Remove currency symbols & spaces
        val_str = re.sub(r'[\$\€\₹\s]', '', val_str)

        # Trailing negative sign
        if val_str.endswith('-'):
            val_str = '-' + val_str[:-1]

        # Handle abbreviations like K, M, B
        match = re.match(r'^([-+]?\d*\.?\d+)([kKmMbB])?$', val_str)
        if match:
            num = float(match.group(1))
            suffix = match.group(2)
            if suffix:
                if suffix.lower() == 'k':
                    num *= 1_000
                elif suffix.lower() == 'm':
                    num *= 1_000_000
                elif suffix.lower() == 'b':
                    num *= 1_000_000_000
            return num

        # Remove thousand separators intelligently
        if val_str.count(',') > 1:
            val_str = val_str.replace(',', '')
        else:
            val_str = val_str.replace(',', '')

        try:
            return float(val_str)
        except ValueError:
            return None

    @staticmethod
    def parse_date(val):
        if pd.isna(val):
            return pd.NaT

        # Excel serial dates check
        if isinstance(val, (int, float)) and 30000 < val < 50000:
            try:
                return datetime(1899, 12, 30) + timedelta(days=int(val))
            except Exception:
                return pd.NaT

        val_str = str(val).strip()

        # Quarters
        q_match = re.match(r'(Q|Quarter)[\s\-]?([1-4])[\s\-]?(\d{2,4})', val_str, re.IGNORECASE)
        if q_match:
            quarter = int(q_match.group(2))
            year = int(q_match.group(3))
            if year < 100:
                year += 2000
            month = 3 * (quarter - 1) + 1
            try:
                return datetime(year, month, 1)
            except Exception:
                return pd.NaT

        # Try common formats
        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%b-%Y', '%b %d, %Y', '%Y/%m/%d', '%d.%m.%Y', '%b %Y', '%B %Y']:
            try:
                return datetime.strptime(val_str, fmt)
            except Exception:
                continue

        try:
            return pd.to_datetime(val_str, errors='coerce')
        except Exception:
            return pd.NaT
