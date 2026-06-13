"""
Script to log benchmark results to MLflow after a model run completes.

This script is called after each benchmark tool finishes running to log
the outputs (predictions, train_time, metrics, curves) to MLflow.

Usage:
    python log_results_to_mlflow.py <task_name> <model_name> <run_type> <benchmark_datasets_dir>

Example:
    python log_results_to_mlflow.py scz random_forest default /path/to/benchmark_datasets/
"""
import sys
import os
import pandas as pd
from pathlib import Path
import mlflow

from sklearn.metrics import (
    roc_auc_score, 
    roc_curve,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    balanced_accuracy_score,
    precision_recall_curve,
    auc
)

# Override with the MLFLOW_TRACKING_URI environment variable to point at a
# shared MLflow tracking server/directory; defaults to a local ./mlflow dir.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "./mlflow")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _save_curve_datapoints(output_dir, filename, dataframe):
    """Helper function to save curve datapoints as CSV artifact."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    curve_path = f"{output_dir}/{filename}"
    dataframe.to_csv(curve_path, index=False)


def calculate_and_log_metrics(y_scores, benchmark_datasets_dir, task_name, output_dir):
    """
    Calculate comprehensive metrics and log them to MLflow.
    
    This function:
    1. Loads ground truth labels
    2. Calculates multiple classification metrics
    3. Generates ROC and PR curve datapoints
    4. Logs all metrics to MLflow
    5. Saves ROC and PR curve data as artifacts
    
    Args:
        predictions: Predicted probabilities (numpy array or similar)
        sample_ids: Sample identifiers corresponding to predictions
        benchmark_datasets_dir: Directory containing test ground truth files
        task_name: Benchmark task name (used to find ground truth file)
        output_dir: Directory to save artifacts
    
    Returns:
        Dictionary of calculated metrics
    """
    # Load ground truth labels
    test_gt_path = f"{benchmark_datasets_dir}/{task_name}_test_gt.csv"
    test_y = pd.read_csv(test_gt_path, index_col=0)
    
    # # Extract true labels and predicted probabilities
    y_true = test_y['label']
    y_pred = y_scores.round()
    
    # Calculate precision-recall curve for PR AUC
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    
    # Calculate all metrics
    metrics_dict = {
        'roc_auc': roc_auc_score(y_true, y_scores),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'pr_auc': auc(recall, precision)
    }
    
    
    # Save ROC curve datapoints
    roc_curve_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': roc_thresholds
    })
    _save_curve_datapoints(output_dir, 'roc_curve.csv', roc_curve_df)
    
    # Save PR curve datapoints
    pr_curve_df = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'threshold': list(pr_thresholds) + [None]  # PR curve has n+1 precision/recall values
    })
    _save_curve_datapoints(output_dir, 'pr_curve.csv', pr_curve_df)
    
    return metrics_dict


def log_model_run(full_output_dir, benchmark_datasets_dir):
    """
    Log a model run to MLflow with predictions, metrics, and artifacts.
    
    This function extracts all necessary information from the output directory structure:
    - task_name, model_name, run_type parsed from directory path
    - predictions loaded from predictions.csv
    - train_time loaded from train_time.txt
    - test metrics calculated from predictions and ground truth
    
    Args:
        full_output_dir: Full path to output directory (e.g., 'benchmarking_outputs/random_forest_scz/default/')
        benchmark_datasets_dir: Directory containing test ground truth files (optional).
                               If not provided, will skip test metrics calculation.
        run_name: Custom run name (optional, auto-generated if not provided)
    
    Returns:
        run_id of the logged run
    """
    # Parse directory structure to extract task_name, model_name, and run_type
    # Expected format: benchmarking_outputs/{model_name}_{task_name}/{run_type}/
    parts = Path(full_output_dir).parts
    model_task = parts[-2]  # e.g., 'random_forest_scz'
    run_type = parts[-1]  # e.g., 'default' or 'optimized'
    
    # Split model_task to get model_name and task_name
    # Assumes format: {task_name}_{model_name}
    task_name, model_name = model_task.split('_', 1)
    mlflow.set_experiment(task_name)
    # Load predictions
    predictions_path = f"{full_output_dir}/predictions.csv"
    predictions_df = pd.read_csv(predictions_path)
    predictions = predictions_df['prediction'].values
    
    # Load train_time
    train_time_path = f"{full_output_dir}/train_time.txt"
    with open(train_time_path, 'r') as f:
        train_time = float(f.read().strip())
    
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        
        # Log basic info
        mlflow.log_param("task_name", task_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("run_type", run_type)

        # Log best parameters from params_search if available
        params_search_path = f"{full_output_dir}/params_search.csv"
        if os.path.exists(params_search_path):
            params_df = pd.read_csv(params_search_path)
            if 'mean_test_score' in params_df.columns:
                best_idx = params_df['mean_test_score'].idxmax()
                best_row = params_df.loc[best_idx]

                metric_cols = {
                    'mean_test_score', 'std_test_score',
                    'mean_fit_time', 'std_fit_time',
                }
                fold_cols = {col for col in params_df.columns if col.startswith('fold_') and col.endswith('_auc')}
                excluded_cols = metric_cols.union(fold_cols)

                for col, value in best_row.items():
                    if col in excluded_cols:
                        continue
                    if pd.isna(value):
                        continue
                    mlflow.log_param(col, value)
        
        # Calculate and log comprehensive test metrics if benchmark_datasets_dir provided
        metrics_dict = calculate_and_log_metrics(
                predictions, 
                benchmark_datasets_dir, 
                task_name, 
                full_output_dir)
        
        metrics_dict['train_time'] = train_time
        
        # Log all metrics to MLflow
        mlflow.log_metrics(metrics_dict)
               
        # Log all files from full_output_dir as artifacts
        mlflow.log_artifacts(full_output_dir)
                
        return run_id

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: python log_results_to_mlflow.py <task_name> <model_name> <run_type> <benchmark_datasets_dir>")
        sys.exit(1)
    
    task_name, model_name, run_type, benchmark_datasets_dir = sys.argv[1:]
    
    # Construct the output directory path
    full_output_dir = f'benchmarking_outputs/{task_name}_{model_name}/{run_type}/'
    
    # Check if output directory exists
    if not os.path.exists(full_output_dir):
        print(f"Error: Output directory does not exist: {full_output_dir}")
        sys.exit(1)
    
    # Check if required files exist
    predictions_file = f"{full_output_dir}/predictions.csv"
    train_time_file = f"{full_output_dir}/train_time.txt"
    
    if not os.path.exists(predictions_file):
        print(f"Error: predictions.csv not found in {full_output_dir}")
        sys.exit(1)
    
    if not os.path.exists(train_time_file):
        print(f"Error: train_time.txt not found in {full_output_dir}")
        sys.exit(1)
    
    # Log to MLflow
    print(f"Logging {task_name} {model_name} {run_type} to MLflow...")
    run_id = log_model_run(full_output_dir, benchmark_datasets_dir)
    print(f"Successfully logged to MLflow with run_id: {run_id}")
