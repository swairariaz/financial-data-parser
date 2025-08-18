#Assignment 2

import pandas as pd

class ExcelProcessor:
    def __init__(self):
        self.files_data = {}

    def load_files(self, files):
        for file in files:
            try:
                xl = pd.ExcelFile(file)
                self.files_data[file] = xl
                print(f"Loaded file: {file} with sheets: {xl.sheet_names}")
            except Exception as e:
                print(f"Error loading {file}: {e}")

    def get_sheet_info(self, file):
        xl = self.files_data.get(file)
        if not xl:
            print(f"No Excel file loaded for {file}")
            return None
        info = {}
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet)
                info[sheet] = {
                    'rows': df.shape[0],
                    'cols': df.shape[1],
                    'columns': list(df.columns)
                }
            except Exception as e:
                print(f"Error parsing sheet {sheet} in {file}: {e}")
        return info

    def extract_data(self, file, sheet_name):
        xl = self.files_data.get(file)
        if not xl:
            print(f"No Excel file loaded for {file}")
            return None
        try:
            df = xl.parse(sheet_name)
            return df
        except Exception as e:
            print(f"Error parsing {sheet_name} in {file}: {e}")
            return None

    def preview_data(self, file, sheet_name, rows=5):
        df = self.extract_data(file, sheet_name)
        if df is not None:
            print(df.head(rows))
        else:
            print(f"Cannot preview data for {sheet_name} in {file}")
