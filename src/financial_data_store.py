import sqlite3
import pandas as pd

class FinancialDataStore:
    def __init__(self, db_path='financial_data.db'):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('PRAGMA foreign_keys = ON;')
        self.conn.commit()

    def sanitize_columns(self, df):
        df = df.copy()
        df.columns = [col.replace('.', '_') for col in df.columns]
        return df

    def add_dataset(self, dataset_name, df, column_types):
        df_copy = self.sanitize_columns(df)

        for col, col_type in column_types.items():
            sanitized_col = col.replace('.', '_')
            if col_type == 'number' and sanitized_col in df_copy.columns:
                df_copy[sanitized_col] = pd.to_numeric(df_copy[sanitized_col], errors='coerce')

                def safe_convert(x):
                    if pd.isna(x):
                        return None
                    try:
                        if isinstance(x, (int, float)):
                            if isinstance(x, float) and x.is_integer():
                                x = int(x)

                            if isinstance(x, int):
                                if -9223372036854775808 <= x <= 9223372036854775807:
                                    return x
                                else:
                                    return str(x)
                            else:
                                return float(x)
                        else:
                            return x
                    except Exception:
                        return str(x)

                df_copy[sanitized_col] = df_copy[sanitized_col].apply(safe_convert)

        try:
            df_copy.to_sql(dataset_name, self.conn, if_exists='replace', index=False)
            print(f"[INFO] Dataset '{dataset_name}' saved to SQLite DB.")
        except Exception as e:
            print(f"[ERROR] Failed to save dataset '{dataset_name}': {e}")

    def query_date_range(self, dataset_name, date_column, start_date, end_date):
        sanitized_date_col = date_column.replace('.', '_')
        query = f"""
            SELECT * FROM {dataset_name}
            WHERE {sanitized_date_col} BETWEEN ? AND ?
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
            return df
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return pd.DataFrame()

    def aggregate_sum(self, dataset_name, group_by_col, sum_col):
        sanitized_group_col = group_by_col.replace('.', '_')
        sanitized_sum_col = sum_col.replace('.', '_')
        query = f"""
            SELECT {sanitized_group_col}, SUM({sanitized_sum_col}) as total_sum
            FROM {dataset_name}
            GROUP BY {sanitized_group_col}
        """
        try:
            df = pd.read_sql_query(query, self.conn)
            if df.empty:
                print("[WARN] Aggregation returned empty result - check column names and data types.")
            return df
        except Exception as e:
            print(f"[ERROR] Aggregation query failed: {e}")
            return pd.DataFrame()
