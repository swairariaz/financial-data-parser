#Assignment 2


import pandas as pd

class FinancialDataStore:
    def __init__(self):
        self.dataframes = {}
        self.metadata = {}
        self.indexes = {}

    def add_dataset(self, name, df, column_types):
        self.dataframes[name] = df
        self.metadata[name] = column_types
        self.create_indexes(name, df, column_types)

    def create_indexes(self, name, df, column_types):
        self.indexes[name] = {}
        for col, typ in column_types.items():
            if typ == 'date':
                try:
                    idx = df.set_index(pd.to_datetime(df[col], errors='coerce'))
                    idx.index.name = col  # Set index name explicitly
                    self.indexes[name]['date_index'] = idx.sort_index()
                except Exception:
                    pass
            elif typ == 'number':
                try:
                    idx = df.set_index(pd.to_numeric(df[col], errors='coerce'))
                    idx.index.name = col  # Set index name explicitly
                    self.indexes[name]['number_index'] = idx.sort_index()
                except Exception:
                    pass

    def query_date_range(self, dataset_name, column, start_date, end_date):
        df = self.dataframes[dataset_name]
        idx = self.indexes.get(dataset_name, {}).get('date_index')
        if idx is not None and idx.index.name == column:
            # Index is already sorted in create_indexes
            filtered = idx.loc[start_date:end_date]
            return filtered.reset_index()
        else:
            df[column] = pd.to_datetime(df[column], errors='coerce')
            mask = (df[column] >= pd.to_datetime(start_date)) & (df[column] <= pd.to_datetime(end_date))
            return df.loc[mask]

    def query_amount_range(self, dataset_name, column, min_amt, max_amt):
        df = self.dataframes[dataset_name]
        idx = self.indexes.get(dataset_name, {}).get('number_index')
        if idx is not None and idx.index.name == column:
            filtered = idx.loc[min_amt:max_amt]
            return filtered.reset_index()
        else:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            mask = (df[column] >= min_amt) & (df[column] <= max_amt)
            return df.loc[mask]

    def query_combined(self, dataset_name, filters: dict):
        df = self.dataframes[dataset_name]
        mask = pd.Series(True, index=df.index)

        for col, (start, end) in filters.items():
            if col not in df.columns:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
                mask &= df[col].between(pd.to_datetime(start), pd.to_datetime(end))
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mask &= df[col].between(start, end)

        return df.loc[mask]

    def aggregate_sum(self, dataset_name, group_by_col, sum_col):
        df = self.dataframes[dataset_name].copy()
        df = df[df[group_by_col].notna() & (df[group_by_col] != '')]

        if pd.api.types.is_datetime64_any_dtype(df[group_by_col]):
            pass
        else:
            if df[group_by_col].dtype == 'object':
                converted = pd.to_datetime(df[group_by_col], errors='coerce', format='%Y-%m-%d')
                if converted.notna().mean() > 0.5:
                    df[group_by_col] = converted
                else:
                    df[group_by_col] = df[group_by_col].astype(str)
            else:
                df[group_by_col] = df[group_by_col].astype(str)

        df[sum_col] = pd.to_numeric(df[sum_col], errors='coerce')
        df_clean = df.dropna(subset=[group_by_col, sum_col])

        print("Sample data before aggregation:")
        print(df_clean[[group_by_col, sum_col]].head(10))
        print(f"Are there duplicates in '{group_by_col}'? {df_clean[group_by_col].duplicated().any()}")
        print(f"Summary stats for '{sum_col}':")
        print(df_clean[sum_col].describe())
        print("----- End of debug info -----")

        result = df_clean.groupby(group_by_col)[sum_col].sum().reset_index()
        return result
