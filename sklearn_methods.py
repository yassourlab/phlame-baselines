from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from benchmark_utils import load_benchmark_data, SEED
import sys
import pandas as pd
import os
import time

# Model configurations
MODELS_DICT = {
    'random_forest': RandomForestClassifier,
    'xgboost': GradientBoostingClassifier,
    'svm': SVC,
    'logistic_regression': LogisticRegression
}

LOGISTIC_REGRESSION_C = [0.01, 0.1, 1, 10, 100]
LOGISTIC_REGRESSION_MAX_ITER = [100000]
LOGISTIC_REGRESSION_CLASS_WEIGHT = [None, 'balanced']

DEFAULT_PARAMS = {
    'svm': {'probability': [True]},
    'logistic_regression': {'max_iter': LOGISTIC_REGRESSION_MAX_ITER}
}


PARAM_GRIDS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 20],
        'criterion': ["gini", "entropy", "log_loss"],
        'max_features': ["sqrt", "log2", 0.2],
        'class_weight': ['balanced', None]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.5],
        'loss': ['log_loss', 'exponential'],
        'max_features': ["sqrt", "log2",None],
        'subsample': [0.8, 1.0]
    },
    'svm': {
        'kernel': ["linear", "rbf", "poly", "sigmoid"],
        'C': [0.01, 0.1, 1, 10],
        'gamma': ["scale", "auto", 0.001],
        'class_weight': ['balanced', None],
        'probability': [True]
    },
    'logistic_regression': [
    # L2 regularization (default) -  5 *5 *2=50
    {
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga',  'newton-cg', 'sag', 'liblinear'],
        'C': LOGISTIC_REGRESSION_C,
        'max_iter': LOGISTIC_REGRESSION_MAX_ITER,
        'class_weight': LOGISTIC_REGRESSION_CLASS_WEIGHT
    },

    # L1 regularization (requires saga solver) - 5 *2*2=20
    {
        'penalty': ['l1'],
        'solver': ['saga','liblinear'],
        'C': LOGISTIC_REGRESSION_C,
        'max_iter': LOGISTIC_REGRESSION_MAX_ITER,
        'class_weight': LOGISTIC_REGRESSION_CLASS_WEIGHT
    },

    # Elastic Net (requires saga solver) - 5 *3 *2=30
    {
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.1, 0.5, 0.9],
        'C': LOGISTIC_REGRESSION_C,
        'max_iter': LOGISTIC_REGRESSION_MAX_ITER,
        'class_weight': LOGISTIC_REGRESSION_CLASS_WEIGHT
    },

    # No regularization (rarely used, but valid)- 1 *4 *2=8
    {
        'penalty': [None],
        'solver': ['lbfgs', 'saga', 'newton-cg', 'sag'], # ‘newton-cg’, ‘sag’, and ‘lbfgs’ 
        'max_iter': LOGISTIC_REGRESSION_MAX_ITER,
        'class_weight': LOGISTIC_REGRESSION_CLASS_WEIGHT
    }
]
}


def run_with_params(train, test_x, model_name, param_grid, run_type, partial_output_dir):
    """
    Run model with given parameter grid using CV paradigm.
    
    Args:
        train: Training data
        test_x: Test features
        model_name: Name of the model to run
        param_grid: Parameter grid for GridSearchCV
        run_type: String identifier for output directory ('default' or 'optimized')
        partial_output_dir: Base output directory
    
    Returns:
        full_output_dir: Path to the output directory
    """
    model_func = MODELS_DICT[model_name]
    
    # Run GridSearchCV with the provided parameter grid
    grid_search = GridSearchCV(
        estimator=model_func(random_state=SEED), 
        param_grid=param_grid,
        scoring='roc_auc', 
        cv=5, 
        n_jobs=-1,
        return_train_score=True,
        refit=False
    )
    grid_search.fit(train.drop(columns=['label']), train.label)
    
    # Get best parameters and train final model on full training set
    best_params = grid_search.cv_results_['params'][grid_search.best_index_]
    best_model = model_func(random_state=SEED, **best_params)
    
    start_time = time.time()
    best_model.fit(train.drop(columns=['label']), train.label)
    time_elapsed = time.time() - start_time
    
    # Save predictions
    full_output_dir = f'{partial_output_dir}/{run_type}/'
    os.makedirs(full_output_dir, exist_ok=True)
    
    predictions = best_model.predict_proba(test_x.drop(columns=['sample_id']))[:, 1]
    predictions_df = pd.DataFrame({
        'sample_id': test_x['sample_id'],
        'prediction': predictions
    })
    predictions_df.to_csv(f'{full_output_dir}/predictions.csv', index=False)

    # Save time_elapsed
    with open(f'{full_output_dir}/train_time.txt', 'w') as f:
        f.write(str(time_elapsed))
    
    # Create CV results CSV
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Extract parameters and flatten them into columns
    params_df = pd.DataFrame(cv_results['params'].tolist())
    
    # Extract fold-level AUC scores
    fold_cols = [col for col in cv_results.columns if col.startswith('split') and col.endswith('_test_score')]
    fold_scores = cv_results[fold_cols]
    fold_scores.columns = [f'fold_{i+1}_auc' for i in range(len(fold_cols))]
    
    # Combine all information
    params_search_df = pd.concat([
        params_df,
        fold_scores,
        cv_results[['mean_test_score', 'std_test_score', 'mean_fit_time', 'std_fit_time']]], axis=1)
    
    params_search_df.to_csv(f'{full_output_dir}/params_search.csv', index=False)

    return full_output_dir


def run_default(train, test_x, model_name, partial_output_dir):
    """Run model with default parameters, using CV paradigm with single parameter set."""
    # Convert default params to grid format (single value per parameter)
    param_grid = DEFAULT_PARAMS.get(model_name, {})
    
    return run_with_params(train, test_x, model_name, param_grid, 'default', partial_output_dir)


def run_optimized(train, test_x, model_name, partial_output_dir):
    """Run model with hyperparameter search, logging all iterations."""
    param_grid = PARAM_GRIDS[model_name]
    
    return run_with_params(train, test_x, model_name, param_grid, 'optimized', partial_output_dir)


if __name__ == '__main__':
    task_name, benchmark_datasets_dir, model_name = sys.argv[1:]
    train, test_x = load_benchmark_data(benchmark_datasets_dir, task_name)
    model_func = MODELS_DICT[model_name]
    default_model_params = DEFAULT_PARAMS.get(model_name, {})
    
    # Define output directory for results
    output_dir = f'benchmarking_outputs/{task_name}_{model_name}'
    
    # Train with default parameters
    print("Training with default parameters...")
    default_output_dir = run_default(train, test_x, model_name, output_dir)
    
    # Train with parameter search
    print("Training with parameter search...")
    optimized_output_dir = run_optimized(train, test_x, model_name, output_dir)


