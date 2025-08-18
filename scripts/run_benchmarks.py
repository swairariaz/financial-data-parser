import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score  # For accuracy in ML/fuzzy; assume scikit-learn is installed

# Import your implemented functions
from core.brute_force import brute_force_subset_sum, direct_match
from core.excel_processor import load_data
from core.ml_processor import dp_subset_sum
from core.fuzzy_matching import fuzzy_match, ga_subset_sum

# Part A: Synthetic Data Generation for Benchmarking
def generate_synthetic_subset_data(size):
    """Generate random transactions and a target sum that is guaranteed to have a subset."""
    transactions = pd.DataFrame({
        'Amount': np.random.randint(1, 100, size=size),
        'Transaction_ID': range(1, size + 1),
        'Description': [f"Test {i}" for i in range(1, size + 1)]
    })
    subset = transactions.sample(n=size // 2, replace=False)
    target = int(round(subset['Amount'].sum()))
    return transactions, target

def generate_synthetic_matching_data(num_transactions, num_targets):
    """Generate synthetic data for matching: amounts + descriptions/refs."""
    amounts = np.random.uniform(10, 1000, num_transactions).round(2)
    descriptions = [f"Invoice #{i:03d}" for i in range(num_transactions)]
    transactions_df = pd.DataFrame({'Amount': amounts, 'Description': descriptions, 'Transaction_ID': range(1, num_transactions + 1)})
    
    target_amounts = amounts[:num_targets] + np.random.normal(0, 5, num_targets)  # Add noise for fuzzy testing
    refs = [f"REF{i:03d}" for i in range(num_targets)]
    targets_df = pd.DataFrame({'Target_Amount': target_amounts.round(2), 'Reference_ID': refs, 'Target_ID': range(1, num_targets + 1)})
    
    ground_truth = [1 if i < num_targets // 2 else 0 for i in range(num_targets)]  # First half are exact matches
    
    return transactions_df, targets_df, ground_truth

# Part B: Benchmark Functions
def benchmark_subset_sum(methods, sizes, runs=5):
    """Benchmark subset sum methods on synthetic data."""
    results = {method: {'times': [], 'accuracies': []} for method in methods}
    for size in sizes:
        for method_name, func in methods.items():
            avg_time = 0
            avg_acc = 0
            for _ in range(runs):
                transactions, target = generate_synthetic_subset_data(size)
                start = time.time()
                if method_name == 'Brute Force':
                    found, matches = func(transactions, target)
                    result = found
                else:
                    result = func(transactions, target)  # Assume others return bool or list
                end = time.time()
                avg_time += (end - start)
                # Accuracy: 1 if found (since guaranteed to exist), else 0
                acc = 1 if result else 0
                if method_name == 'Brute Force' and matches:
                    acc = 1 if abs(matches[0]['Subset_Sum'] - target) < 1e-6 else 0
                elif isinstance(result, list) and result:
                    acc = 1 if abs(sum(result) - target) < 1e-6 else 0
                avg_acc += acc
            results[method_name]['times'].append(avg_time / runs)
            results[method_name]['accuracies'].append(avg_acc / runs)
    return results

def benchmark_matching(methods, sizes, runs=5):
    """Benchmark matching methods on synthetic data."""
    results = {method: {'times': [], 'accuracies': []} for method in methods}
    for size in sizes:
        num_transactions = size
        num_targets = size // 2
        for method_name, func in methods.items():
            avg_time = 0
            avg_acc = 0
            for _ in range(runs):
                transactions_df, targets_df, ground_truth = generate_synthetic_matching_data(num_transactions, num_targets)
                start = time.time()
                matches = func(transactions_df, targets_df)  # Returns list of match dictionaries
                end = time.time()
                avg_time += (end - start)
                # Accuracy: Use accuracy_score with thresholded predictions
                predictions = [0] * num_targets
                if method_name == 'Fuzzy Matching':
                    # Extract Similarity_Score from each match dictionary and threshold
                    for match in matches:
                        if 'Target_ID' in match and match['Target_ID'] <= num_targets:
                            idx = match['Target_ID'] - 1
                            predictions[idx] = 1 if match.get('Similarity_Score', float('inf')) <= 5000 else 0  # Use actual threshold
                else:  # Direct Matching
                    predictions = [1 if any(m['Transaction_ID'] == tid for m in matches) else 0 for tid in transactions_df['Transaction_ID'][:num_targets]]
                acc = accuracy_score(ground_truth, predictions[:len(ground_truth)])
                avg_acc += acc
            results[method_name]['times'].append(avg_time / runs)
            results[method_name]['accuracies'].append(avg_acc / runs)
    return results

# Part C: Run Benchmarks (Task 5.1)
subset_sizes = [5, 10, 15, 20, 25]  # Small for brute force
matching_sizes = [50, 100, 200, 500, 1000]  # Larger, as matching is faster

subset_methods = {
    'Brute Force': brute_force_subset_sum,
    'Dynamic Programming': dp_subset_sum,
    'Genetic Algorithm': ga_subset_sum
}

matching_methods = {
    'Direct Matching': direct_match,
    'Fuzzy Matching': fuzzy_match
}

subset_results = benchmark_subset_sum(subset_methods, subset_sizes)
matching_results = benchmark_matching(matching_methods, matching_sizes)

# Save to CSV for reporting
subset_df = pd.DataFrame({k: v['times'] for k, v in subset_results.items()}, index=subset_sizes)
subset_df.columns = [f"{col} Time (s)" for col in subset_df.columns]
subset_df_acc = pd.DataFrame({k: v['accuracies'] for k, v in subset_results.items()}, index=subset_sizes)
subset_df_acc.columns = [f"{col} Accuracy" for col in subset_df_acc.columns]
subset_combined = pd.concat([subset_df, subset_df_acc], axis=1)
subset_combined.to_csv('data/processed/subset_benchmark.csv')

matching_df = pd.DataFrame({k: v['times'] for k, v in matching_results.items()}, index=matching_sizes)
matching_df.columns = [f"{col} Time (s)" for col in matching_df.columns]
matching_df_acc = pd.DataFrame({k: v['accuracies'] for k, v in matching_results.items()}, index=matching_sizes)
matching_df_acc.columns = [f"{col} Accuracy" for col in matching_df_acc.columns]
matching_combined = pd.concat([matching_df, matching_df_acc], axis=1)
matching_combined.to_csv('data/processed/matching_benchmark.csv')

print("Subset Sum Benchmarks:\n", subset_combined)
print("\nMatching Benchmarks:\n", matching_combined)

# Part D: Visualizations (Task 5.2)
# Time vs Size for Subset Sum
plt.figure(figsize=(10, 6))
for method, data in subset_results.items():
    plt.plot(subset_sizes, data['times'], label=method)
plt.xlabel('Number of Transactions')
plt.ylabel('Average Execution Time (s)')
plt.title('Subset Sum Methods: Time vs Dataset Size')
plt.legend()
plt.grid(True)
plt.savefig('data/processed/subset_time_plot.png')

# Accuracy Bar Chart for Subset Sum
plt.figure(figsize=(10, 6))
width = 0.2
for i, (method, data) in enumerate(subset_results.items()):
    plt.bar(np.array(subset_sizes) + i*width, data['accuracies'], width, label=method)
plt.xlabel('Number of Transactions')
plt.ylabel('Average Accuracy')
plt.title('Subset Sum Methods: Accuracy vs Dataset Size')
plt.legend()
plt.grid(True)
plt.savefig('data/processed/subset_accuracy_plot.png')

# Similar for Matching
plt.figure(figsize=(10, 6))
for method, data in matching_results.items():
    plt.plot(matching_sizes, data['times'], label=method)
plt.xlabel('Number of Transactions')
plt.ylabel('Average Execution Time (s)')
plt.title('Matching Methods: Time vs Dataset Size')
plt.legend()
plt.grid(True)
plt.savefig('data/processed/matching_time_plot.png')

plt.figure(figsize=(10, 6))
for i, (method, data) in enumerate(matching_results.items()):
    plt.bar(np.array(matching_sizes) + i*width, data['accuracies'], width, label=method)
plt.xlabel('Number of Transactions')
plt.ylabel('Average Accuracy')
plt.title('Matching Methods: Accuracy vs Dataset Size')
plt.legend()
plt.grid(True)
plt.savefig('data/processed/matching_accuracy_plot.png')

# Histogram of Fuzzy Matching Scores
if os.path.exists('data/processed/fuzzy_matches.csv'):
    fuzzy_df = pd.read_csv('data/processed/fuzzy_matches.csv')
    plt.figure(figsize=(10, 6))
    plt.hist(fuzzy_df['Similarity_Score'], bins=20, color='purple', edgecolor='black')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Fuzzy Matching Similarity Scores')
    plt.grid(True)
    plt.savefig('data/processed/fuzzy_scores_histogram.png')