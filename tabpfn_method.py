"""TabPFN Benchmarking Script

REQUIREMENTS:
- Hugging Face token with access to gated model 'Prior-Labs/tabpfn_2_5'
- GPU with CUDA support

SETUP INSTRUCTIONS:
1. Create a Hugging Face account at https://huggingface.co/
2. Create a read token at https://huggingface.co/settings/tokens
3. Accept the gated model terms at https://huggingface.co/Prior-Labs/tabpfn_2_5
4. Set the token in your environment:
   
   Option A (recommended): Add to ~/.bashrc
   echo 'export HF_TOKEN="hf_YourTokenHere"' >> ~/.bashrc
   source ~/.bashrc
   
   Option B: Login via CLI
   huggingface-cli login
   
   See https://docs.priorlabs.ai/how-to-access-gated-models for details.
"""

from tabpfn import TabPFNClassifier
from benchmark_utils import load_benchmark_data, SEED
from cv_splitters import build_stratified_group_cv
import sys
import pandas as pd
import os
import time
import warnings
import torch

# Suppress TabPFN warnings for cleaner output
warnings.filterwarnings('ignore')

# TabPFN hyperparameters for grid search
# Note: TabPFN is a pre-trained model, so hyperparameter search is limited
DEFAULT_PARAMS = {
    'n_estimators': [8],  # Default ensemble size 
    'device': ['cuda'],  # Use GPU
    'ignore_pretraining_limits': [True],
}

PARAM_GRIDS = {
    'n_estimators': [1, 2, 4, 8, 16, 32],  # Ensemble size affects prediction quality vs speed
    'balance_probabilities': [True, False],  # Whether to balance probabilities (can affect performance)
    'ignore_pretraining_limits': [True],  # Whether to ignore pretraining limits (can improve performance but may cause issues)
    'device': ['cuda'],  # Always use GPU
}


def run_with_params(train, test_x, train_subject_ids, param_grid, run_type, partial_output_dir):
    """
    Run TabPFN with given parameter grid using CV paradigm.
    
    Args:
        train: Training data
        test_x: Test features
        param_grid: Parameter grid dict with lists of parameter values
        run_type: String identifier for output directory ('default' or 'optimized')
        partial_output_dir: Base output directory
    
    Returns:
        full_output_dir: Path to the output directory
    """
    from sklearn.model_selection import cross_val_score
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    param_combinations = [
        dict(zip(param_names, v)) 
        for v in product(*param_values)
    ]
    
    print(f"Testing {len(param_combinations)} parameter combination(s)...")
    
    # Prepare base training data
    X_train_base = train.drop(columns=['label']).values
    y_train = train['label'].values
    X_test_base = test_x.drop(columns=['sample_id']).values
    grouped_cv = build_stratified_group_cv(n_splits=5, random_state=SEED)
    
    # Perform cross-validation for all combinations
    results = []
    
    for params in param_combinations:
        params = params.copy()
        print(f"  Testing {params}...")
        
        # Create classifier with current params
        clf = TabPFNClassifier(
            n_estimators=params.get('n_estimators', 8),
            device=params.get('device', 'cuda'),
            ignore_pretraining_limits=params.get('ignore_pretraining_limits', True),
            random_state=SEED
        )
        
        # Perform 5-fold CV
        try:
            fold_start = time.time()
            cv_scores = cross_val_score(
                clf,
                X_train_base,
                y_train,
                groups=train_subject_ids,
                cv=grouped_cv,
                scoring='roc_auc',
                n_jobs=1,
            )
            fold_time = time.time() - fold_start
            
            # Build result row
            result = params.copy()
            result['mean_test_score'] = cv_scores.mean()
            result['std_test_score'] = cv_scores.std()
            result['mean_fit_time'] = fold_time / 5
            result['std_fit_time'] = 0  # Not tracking individual fold times
            
            # Add individual fold scores
            for i, score in enumerate(cv_scores):
                result[f'fold_{i+1}_auc'] = score
            
            results.append(result)
            print(f"    Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Find best parameters
    if not results:
        raise ValueError("No valid parameter combinations found")
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['mean_test_score'].idxmax()
    best_params = {k: results_df.loc[best_idx, k] for k in param_names}
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV ROC-AUC: {results_df.loc[best_idx, 'mean_test_score']:.4f}")
    
    # Train final model on full training data with best parameters
    final_clf = TabPFNClassifier(
        n_estimators=best_params.get('n_estimators', 8),
        device=best_params.get('device', 'cuda'),
        ignore_pretraining_limits=best_params.get('ignore_pretraining_limits', True),
        random_state=SEED
    )
    
    start_time = time.time()
    final_clf.fit(X_train_base, y_train)
    
    # Predict on test set
    predictions = final_clf.predict_proba(X_test_base)[:, 1]
    time_elapsed = time.time() - start_time
    
    # Save outputs
    full_output_dir = f'{partial_output_dir}/{run_type}/'
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'sample_id': test_x['sample_id'],
        'prediction': predictions
    })
    predictions_df.to_csv(f'{full_output_dir}/predictions.csv', index=False)
    
    # Save training time
    with open(f'{full_output_dir}/train_time.txt', 'w') as f:
        f.write(str(time_elapsed))
    
    # Save parameter search results
    results_df.to_csv(f'{full_output_dir}/params_search.csv', index=False)
    
    return full_output_dir


def run_default(train, test_x, train_subject_ids, partial_output_dir):
    """Run TabPFN with default parameters, using CV paradigm with single parameter set."""
    print("Running TabPFN with default parameters...")
    
    return run_with_params(
        train,
        test_x,
        train_subject_ids,
        DEFAULT_PARAMS,
        'default',
        partial_output_dir,
    )


def run_optimized(train, test_x, train_subject_ids, partial_output_dir):
    """Run TabPFN with hyperparameter search."""
    print("Running TabPFN with hyperparameter search...")
    
    return run_with_params(
        train,
        test_x,
        train_subject_ids,
        PARAM_GRIDS,
        'optimized',
        partial_output_dir,
    )


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python tabpfn_method.py <task_name> <benchmark_datasets_dir>")
        sys.exit(1)
    
    task_name, benchmark_datasets_dir = sys.argv[1:]
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Loading data for task: {task_name}")
    train, test_x, train_subject_ids = load_benchmark_data(
        benchmark_datasets_dir,
        task_name,
        return_subject_ids=True,
    )
    
    # Define output directory for results
    output_dir = f'benchmarking_outputs/{task_name}_tabpfn'
    
    # Train with default parameters
    print("\n" + "="*60)
    print("DEFAULT RUN")
    print("="*60)
    default_output_dir = run_default(train, test_x, train_subject_ids, output_dir)
    print(f"✓ Default run completed. Results saved to: {default_output_dir}")
    
    # Train with parameter search
    print("\n" + "="*60)
    print("OPTIMIZED RUN")
    print("="*60)
    optimized_output_dir = run_optimized(train, test_x, train_subject_ids, output_dir)
    print(f"✓ Optimized run completed. Results saved to: {optimized_output_dir}")
    
    print("\n" + "="*60)
    print("ALL RUNS COMPLETED SUCCESSFULLY")
    print("="*60)
