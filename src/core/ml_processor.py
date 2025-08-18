# Assignment 3

import pandas as pd
import time

# Task 3.1: Feature Engineering 
print("\nPart 3: Machine Learning Approach")
print("Task 3.1: Feature Engineering")

# Load cleaned data
df_transactions = pd.read_csv('data/processed/clean_transactions.csv')
df_targets = pd.read_csv('data/processed/clean_targets.csv')
df_direct_matches = pd.read_csv('data/processed/direct_matches.csv')

# Create all possible transaction-target pairs (sample 1000 for efficiency)
pairs = []
for idx_trans, trans in df_transactions.sample(n=1000, random_state=42).iterrows():
    for idx_target, target in df_targets.sample(n=1000, random_state=42).iterrows():
        pairs.append({
            'Transaction_ID': trans['Transaction_ID'],
            'Target_ID': target['Target_ID'],
            'Amount': trans['Amount'],
            'Target_Amount': target['Target_Amount'],
            'Description': trans['Description']
        })

df_pairs = pd.DataFrame(pairs)

# Engineer features
df_pairs['Amount_Diff'] = abs(df_pairs['Amount'] - df_pairs['Target_Amount'])
df_pairs['Amount_Ratio'] = df_pairs.apply(
    lambda row: row['Amount'] / row['Target_Amount'] if row['Target_Amount'] != 0 else float('inf'), axis=1
)
df_pairs['Desc_Length'] = df_pairs['Description'].str.len()
df_pairs['Desc_Contains_Invoice'] = df_pairs['Description'].str.contains('Invoice', case=False).astype(int)

# Label based on direct matches
df_pairs['Is_Direct_Match'] = 0
for idx, match in df_direct_matches.iterrows():
    mask = (df_pairs['Transaction_ID'] == match['Transaction_ID']) & (df_pairs['Target_ID'] == match['Target_ID'])
    df_pairs.loc[mask, 'Is_Direct_Match'] = 1

# Output sample features
print("Sample engineered features:")
print(df_pairs[['Transaction_ID', 'Target_ID', 'Amount_Diff', 'Amount_Ratio', 'Desc_Length', 'Desc_Contains_Invoice', 'Is_Direct_Match']].head())

# Save feature-engineered data
df_pairs.to_csv('data/processed/feature_engineered_data.csv', index=False)
print("Saved feature-engineered data to 'data/processed/feature_engineered_data.csv'.")

# Task 3.2: Dynamic Programming Enhancement 
def dp_subset_sum(transactions, target_amount):
    """Find a subset of transactions that sum to the target amount using dynamic programming."""
    amounts = [int(round(amount)) for amount in transactions['Amount'].tolist()]
    transaction_ids = transactions['Transaction_ID'].tolist()
    target_amount = int(round(target_amount))  # Round to nearest integer
    n = len(amounts)
    dp = [[False] * (target_amount + 1) for _ in range(n + 1)]
    dp[0][0] = True  # Base case: sum 0 is achievable with 0 amounts
    keep = [[0] * (target_amount + 1) for _ in range(n + 1)]  # To reconstruct solution

    for i in range(1, n + 1):
        dp[i] = dp[i-1].copy()
        for s in range(target_amount + 1):
            if dp[i-1][s]:
                dp[i][s] = True
                keep[i][s] = 0  # No new amount used
            if s >= amounts[i-1] and dp[i-1][s - amounts[i-1]]:
                dp[i][s] = True
                keep[i][s] = 1  # Use the current amount

    # Reconstruct the solution
    if dp[n][target_amount]:
        selected_ids = []
        s = target_amount
        for i in range(n, 0, -1):
            if keep[i][s] == 1:
                selected_ids.append(transaction_ids[i-1])
                s -= amounts[i-1]
        return True, [{
            'Subset_Sum': target_amount,
            'Transaction_IDs': selected_ids
        }]
    return False, []

print("\nTask 3.2: Dynamic Programming Enhancement")

# Load cleaned data and sample for feasibility
df_transactions = pd.read_csv('data/processed/clean_transactions.csv').head(100)  # Reduced to 100
df_targets = pd.read_csv('data/processed/clean_targets.csv').head(20)  # Reduced to 20
df_targets = df_targets[df_targets['Target_Amount'] <= 100000]  # Limit target amount

# Initialize list to store DP matches
dp_matches = []

# Start timing
start_time = time.time()

# Dynamic Programming for subset sum
for idx_target, row_target in df_targets.iterrows():
    target_amount = row_target['Target_Amount']
    target_id = row_target['Target_ID']
    found, matches = dp_subset_sum(df_transactions, target_amount)
    if found:
        for match in matches:
            dp_matches.append({
                'Target_ID': target_id,
                'Target_Amount': target_amount,
                'Subset_Sum': match['Subset_Sum'],
                'Transaction_IDs': match['Transaction_IDs']
            })

# End timing
end_time = time.time()
execution_time = end_time - start_time

# Convert to DataFrame
dp_matches_df = pd.DataFrame(dp_matches) if dp_matches else pd.DataFrame(columns=['Target_ID', 'Target_Amount', 'Subset_Sum', 'Transaction_IDs'])

# Output results
print(f"Number of DP subset matches found: {len(dp_matches_df)}")
if not dp_matches_df.empty:
    print("Sample DP subset matches:")
    print(dp_matches_df.head())
else:
    print("No DP subset matches found.")
print(f"Execution time for Task 3.2: {execution_time:.2f} seconds")

# Save matches
dp_matches_df.to_csv('data/processed/dp_matches.csv', index=False)
print("Saved DP matches to 'data/processed/dp_matches.csv'.")