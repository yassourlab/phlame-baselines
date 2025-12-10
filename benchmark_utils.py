"""
Shared utilities for benchmark experiments (non-MLflow functions).
"""
import pandas as pd

SEED = 17
TRAIN_COLS_TO_DROP = ['sample_id', 'subject_id', 'study_id']
TEST_COLS_TO_DROP = ['sample_id']


def load_benchmark_data(benchmark_datasets_dir, task_name):
    """Load train and test datasets for a benchmark task."""
    train_path = f"{benchmark_datasets_dir}/{task_name}_train.csv"
    test_x_path = f"{benchmark_datasets_dir}/{task_name}_test.csv"
    
    train = pd.read_csv(train_path).drop(columns=TRAIN_COLS_TO_DROP)
    test_x = pd.read_csv(test_x_path)
    
    return train, test_x
