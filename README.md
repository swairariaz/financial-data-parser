📊 Financial Data Reconciliation Project

A robust system for reconciling financial transactions across datasets using both classic and modern algorithmic approaches.
This project integrates data preparation, brute-force methods, machine learning techniques, advanced heuristics, and performance benchmarking — all with modular, professional-grade Python code.

🚀 Features

Data Handling → Load, clean, and standardize financial records.

Matching Algorithms → Exact matching, subset sum (brute force, DP), genetic algorithms.

Advanced Matching → Fuzzy reconciliation with similarity scoring.

Performance Analysis → Benchmark accuracy & execution time.

Visualizations → Comparative charts of algorithm performance.

📝 Tasks
Part 1: Data Preparation & Exploration

Load & inspect datasets (info, duplicates).

Handle missing values & standardize formats.

Create unique transaction identifiers.
Implemented in excel_processor.py

Part 2: Brute Force Approach

Exact match by transaction amounts.

Subset sum via brute force.

Execution time measurement.
Implemented in brute_force.py

Part 3: Machine Learning Approach

Feature engineering (differences, text features).

Dynamic programming for subset sum.

Optional ML models for prediction.
Implemented in ml_processor.py

Part 4: Advanced Techniques

Genetic algorithms for efficient matching.

Fuzzy matching with similarity scores.
Implemented in fuzzy_matching.py

Part 5: Performance Comparison

Benchmark accuracy & execution time.

Generate visualization reports.
Implemented in run_benchmarks.py

⚙️ Dependencies

Python 3.x

pandas, numpy, matplotlib

scikit-learn

python-Levenshtein

Standard libs: itertools, time, random

📦 Install everything with:

pip install pandas numpy matplotlib scikit-learn python-Levenshtein

▶️ How to Run

Setup Virtual Environment

python -m venv venv
source venv/bin/activate   # Unix/Mac
venv\Scripts\activate      # Windows


Install Dependencies

pip install -r requirements.txt


(or run pip install command above)

Execute Tasks

# Data Preparation
python src/core/excel_processor.py

# Brute Force
python src/core/brute_force.py

# ML Approach
python src/core/ml_processor.py

# Advanced Techniques
python src/core/fuzzy_matching.py


Run Benchmarks

python scripts/run_benchmarks.py


Generates performance metrics & visualizations.

🗒️ Notes

All scripts print progress and results.

Outputs are saved for traceability.

You can adjust dataset sizes in run_benchmarks.py.