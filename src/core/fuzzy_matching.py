# Assignment 4

import random
from Levenshtein import distance as levenshtein_distance
import pandas as pd
import time

# Task 4.1: Genetic Algorithm for Subset Selection 
def ga_subset_sum(transactions, target_amount):
    """Find subsets of transactions that sum to the target amount using genetic algorithm."""
    amounts = [int(round(amount)) for amount in transactions['Amount'].tolist()]
    transaction_ids = transactions['Transaction_ID'].tolist()
    population_size = 50
    generations = 50
    mutation_rate = 0.2

    def fitness(individual):
        return abs(sum(amounts[i] for i in individual if i < len(amounts)) - target_amount)

    def create_individual():
        return [random.randint(0, len(amounts)-1) for _ in range(random.randint(1, 3))]  # Limit to 3 items

    def crossover(parent1, parent2):
        if not parent1 or not parent2:
            return parent1 or parent2
        point = random.randint(0, min(len(parent1), len(parent2)))
        return parent1[:point] + [x for x in parent2 if x not in parent1[:point]]

    def mutate(individual):
        if random.random() < mutation_rate and individual:
            idx = random.randint(0, len(individual)-1)
            individual[idx] = random.randint(0, len(amounts)-1)
        return individual

    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    best_fitness = float('inf')

    for _ in range(generations):
        # Evaluate fitness
        fitness_scores = [fitness(ind) for ind in population if ind]
        if not fitness_scores:
            break
        # Replace infinite or zero weights with a small positive value
        weights = [1 / max(f, 1e-10) for f in fitness_scores]  # Avoid division by zero
        best_idx = fitness_scores.index(min(fitness_scores))
        if fitness_scores[best_idx] < best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_individual = population[best_idx]

        # Selection, crossover, mutation
        new_population = []
        for _ in range(population_size // 2):
            if not population or not weights:
                break
            try:
                parent1 = random.choices(population, weights=weights, k=1)[0]
                parent2 = random.choices(population, weights=weights, k=1)[0]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                new_population.extend([mutate(child1), mutate(child2)])
            except ValueError:
                # Fallback to random selection if weights fail
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                new_population.extend([mutate(child1), mutate(child2)])
        population = new_population[:population_size]

    if best_fitness < 1000:  # Relaxed threshold
        selected_ids = [transaction_ids[i] for i in best_individual if i < len(transaction_ids)]
        return True, [{
            'Subset_Sum': sum(amounts[i] for i in best_individual if i < len(amounts)),
            'Transaction_IDs': selected_ids
        }]
    return False, []

print("\nPart 4: Advanced Techniques")
print("Task 4.1: Genetic Algorithm for Subset Selection")

# Load and sample data
df_transactions = pd.read_csv('data/processed/clean_transactions.csv').head(50)
df_targets = pd.read_csv('data/processed/clean_targets.csv').head(10)
df_targets = df_targets[df_targets['Target_Amount'] <= 50000]  # Smaller target range

# Initialize list to store GA matches
ga_matches = []

# Start timing
start_time = time.time()

for idx_target, row_target in df_targets.iterrows():
    target_amount = int(round(row_target['Target_Amount']))
    target_id = row_target['Target_ID']
    found, matches = ga_subset_sum(df_transactions, target_amount)
    if found:
        for match in matches:
            ga_matches.append({
                'Target_ID': target_id,
                'Target_Amount': target_amount,
                'Subset_Sum': match['Subset_Sum'],
                'Transaction_IDs': match['Transaction_IDs']
            })

# End timing
end_time = time.time()
execution_time = end_time - start_time

ga_matches_df = pd.DataFrame(ga_matches) if ga_matches else pd.DataFrame(columns=['Target_ID', 'Target_Amount', 'Subset_Sum', 'Transaction_IDs'])

print(f"Number of GA subset matches found: {len(ga_matches_df)}")
if not ga_matches_df.empty:
    print("Sample GA subset matches:")
    print(ga_matches_df.head())
else:
    print("No GA subset matches found.")
print(f"Execution time for Task 4.1: {execution_time:.2f} seconds")
ga_matches_df.to_csv('data/processed/ga_matches.csv', index=False)
print("Saved GA matches to 'data/processed/ga_matches.csv'.")

# Task 4.2: Fuzzy Matching with Similarity Scores 
def fuzzy_match(transactions_df, targets_df):
    """Find fuzzy matches between transactions and targets based on amount and description similarity."""
    fuzzy_matches = []
    # Add a dummy description to targets based on Reference_ID for fuzzy matching
    targets_df = targets_df.copy()
    targets_df['Description'] = targets_df['Reference_ID'].astype(str) + "_REF"

    for idx_target, target in targets_df.iterrows():
        target_id = target['Target_ID']
        target_desc = target['Description']
        best_score = float('inf')
        best_trans = None

        for idx_trans, trans in transactions_df.iterrows():
            trans_desc = trans['Description']
            amount_diff = abs(trans['Amount'] - target['Target_Amount'])
            desc_dist = levenshtein_distance(trans_desc, target_desc) if trans_desc and target_desc else float('inf')
            score = amount_diff + desc_dist * 100  # Adjusted weight

            if score < best_score:
                best_score = score
                best_trans = trans

        if best_score < 5000:  # Lowered threshold
            fuzzy_matches.append({
                'Target_ID': target_id,
                'Target_Amount': target['Target_Amount'],
                'Transaction_ID': best_trans['Transaction_ID'],
                'Amount': best_trans['Amount'],
                'Description': best_trans['Description'],
                'Similarity_Score': best_score
            })
    return fuzzy_matches

print("\nTask 4.2: Fuzzy Matching with Similarity Scores")

# Load and sample data
df_transactions = pd.read_csv('data/processed/clean_transactions.csv').head(100)
df_targets = pd.read_csv('data/processed/clean_targets.csv').head(20)

# Execute fuzzy matching
fuzzy_matches = fuzzy_match(df_transactions, df_targets)

fuzzy_matches_df = pd.DataFrame(fuzzy_matches) if fuzzy_matches else pd.DataFrame(columns=['Target_ID', 'Target_Amount', 'Transaction_ID', 'Amount', 'Description', 'Similarity_Score'])

print(f"Number of fuzzy matches found: {len(fuzzy_matches_df)}")
if not fuzzy_matches_df.empty:
    print("Sample fuzzy matches:")
    print(fuzzy_matches_df.head())
else:
    print("No fuzzy matches found.")
print(f"Execution time for Task 4.2: {execution_time:.2f} seconds")
fuzzy_matches_df.to_csv('data/processed/fuzzy_matches.csv', index=False)
print("Saved fuzzy matches to 'data/processed/fuzzy_matches.csv'.")