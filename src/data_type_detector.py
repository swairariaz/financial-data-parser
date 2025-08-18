#Assignment 2

import pandas as pd
from dateutil.parser import parse as date_parse
import re

def detect_column_type(series, sample_size=100):
    """
    Detect if the column is date, number, or string with confidence.
    Returns: (detected_type, confidence)
    detected_type: 'date', 'number', or 'string'
    confidence: float between 0 and 1
    """
    sample = series.dropna().head(sample_size).astype(str)
    if sample.empty:
        return 'string', 0.0

    # Check date parse success ratio
    date_success = 0
    for val in sample:
        try:
            date_parse(val, fuzzy=False)
            date_success += 1
        except:
            pass
    date_confidence = date_success / len(sample)
    if date_confidence > 0.7:
        return 'date', date_confidence

    # Check number parse success ratio (handle commas, currency, negatives)
    number_success = 0
    for val in sample:
        val_clean = re.sub(r'[^\d\.\-\(\)KMkmbB]', '', val)  # Remove letters except K,M,B and symbols
        val_clean = val_clean.replace('(', '-').replace(')', '')  # Convert (123) to -123
        val_clean = val_clean.rstrip('-')  # Remove trailing minus if any

        # Remove K,M,B and test parse after conversion later
        try:
            float(val_clean)
            number_success += 1
        except:
            # try removing suffix and parse
            if val_clean[:-1].replace('.', '').isdigit():
                number_success += 1
            else:
                pass
    number_confidence = number_success / len(sample)
    if number_confidence > 0.7:
        return 'number', number_confidence

    return 'string', max(date_confidence, number_confidence)
