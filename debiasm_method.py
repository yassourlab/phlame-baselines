"""DEBIAS-M Benchmarking Script

Uses OnlineDebiasMClassifier to avoid data leakage. This classifier handles
previously unobserved batches at test time via an online inference step.

DEBIAS-M expects the first column of X to be batch IDs (non-negative integers).
Training data uses study_id as batch; test data (which lacks study_id) is assigned
a single new batch ID so the online step can infer its biases.

Usage:
    python debiasm_method.py <task_name> <benchmark_datasets_dir>
    python debiasm_method.py <task_name> <benchmark_datasets_dir> --mode list-configs --run-type <default|optimized>
    python debiasm_method.py <task_name> <benchmark_datasets_dir> --mode run-config --run-type <default|optimized> --config-index <N>
    python debiasm_method.py <task_name> <benchmark_datasets_dir> --mode aggregate --run-type <default|optimized>
"""

from debiasm import OnlineDebiasMClassifier
from benchmark_utils import load_benchmark_data, SEED
from cv_splitters import build_stratified_group_cv
from itertools import product
import argparse
import contextlib
import json
import numpy as np
import pandas as pd
import sys
import os
import time


# Default parameters (single config for baseline CV run)
BASE_DEFAULT_PARAMS = {
    'learning_rate': [0.005],
    'min_epochs': [25],
    'l2_strength': [0],
    'w_l2': [0],
}

# Optimized parameter grid (4×3×3×3 = 108 combinations)
BASE_PARAM_GRIDS = {
    'learning_rate': [0.0005, 0.001, 0.005, 0.01],
    'min_epochs': [10, 25, 40],
    'l2_strength': [0, 0.01, 0.1],
    'w_l2': [0, 0.01, 0.1],
}

DEFAULT_PARAMS = {k: list(v) for k, v in BASE_DEFAULT_PARAMS.items()}
PARAM_GRIDS = {k: list(v) for k, v in BASE_PARAM_GRIDS.items()}

# Keep features with train-set prevalence above threshold at the given abundance.
FEATURE_SELECTION_ABUNDANCE_THRESHOLD = 0.001
FEATURE_SELECTION_PREVALENCE_THRESHOLD = 0.001


def apply_feature_selection(X_train, X_test,
                            abundance_threshold=FEATURE_SELECTION_ABUNDANCE_THRESHOLD,
                            prevalence_threshold=FEATURE_SELECTION_PREVALENCE_THRESHOLD):
    """Select features where fraction(values > abundance_threshold) > prevalence_threshold.

    Expects the first column to be batch IDs and keeps it unchanged.
    """
    train_features = X_train[:, 1:]
    prevalence = (train_features > abundance_threshold).mean(axis=0)
    selected_feature_mask = prevalence > prevalence_threshold
    selected_feature_count = int(selected_feature_mask.sum())
    total_feature_count = train_features.shape[1]

    print(
        "Applying feature selection: "
        f"abundance>{abundance_threshold}, prevalence>{prevalence_threshold}."
    )
    print(
        f"Selected {selected_feature_count}/{total_feature_count} features "
        f"({selected_feature_count / total_feature_count:.2%})."
    )

    if selected_feature_count == 0:
        raise ValueError(
            "Feature selection removed all features. "
            "Please relax abundance/prevalence thresholds."
        )

    selected_cols = np.concatenate(([0], np.where(selected_feature_mask)[0] + 1))
    return X_train[:, selected_cols], X_test[:, selected_cols]


def add_epsilon_to_zero_sum_rows(X, epsilon=1e-12):
    """Add epsilon to feature values for rows whose feature-sum is 0."""
    feature_sums = X[:, 1:].sum(axis=1)
    zero_sum_mask = np.isclose(feature_sums, 0.0)

    if np.any(zero_sum_mask):
        X = X.copy()
        X[zero_sum_mask, 1:] += epsilon
        print(
            f"Adjusted {zero_sum_mask.sum()} sample(s) with zero feature-sum "
            f"by adding epsilon={epsilon}."
        )

    return X


def run_with_params(X_train, y_train, X_test, test_sample_ids, train_subject_ids,
                    param_grid, run_type, partial_output_dir):
    """
    Run DEBIAS-M with given parameter grid using manual 5-fold stratified CV.

    Args:
        X_train: Training features with batch column prepended
        y_train: Training labels
        X_test: Test features with batch column prepended
        test_sample_ids: Sample IDs for prediction output
        param_grid: Dict of parameter lists to search over
        run_type: 'default' or 'optimized'
        partial_output_dir: Base output directory

    Returns:
        full_output_dir: Path to the output directory
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    param_combinations = [
        dict(zip(param_names, v))
        for v in product(*param_values)
    ]

    print(f"Testing {len(param_combinations)} parameter combination(s)...")

    grouped_cv = build_stratified_group_cv(n_splits=5, random_state=SEED)
    results = []

    X_train_prepared, X_test_prepared = apply_feature_selection(X_train, X_test)
    X_train_prepared = add_epsilon_to_zero_sum_rows(X_train_prepared)
    X_test_prepared = add_epsilon_to_zero_sum_rows(X_test_prepared)

    for params in param_combinations:
        params = params.copy()
        print(f"  Testing {params}...")

        fold_scores = []
        fold_times = []

        
        for fold_idx, (train_idx, val_idx) in enumerate(
            grouped_cv.split(X_train_prepared, y_train, train_subject_ids)
        ):
            X_fold_train = X_train_prepared[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train_prepared[val_idx]
            y_fold_val = y_train[val_idx]

            clf = OnlineDebiasMClassifier(
                batch_str=params['batch_str'],
                learning_rate=params['learning_rate'],
                min_epochs=params['min_epochs'],
                l2_strength=params['l2_strength'],
                w_l2=params['w_l2'],
                random_state=SEED,
            )

            fold_start = time.time()
            clf.fit(X_fold_train, y_fold_train)
            proba = clf.predict_proba(X_fold_val)[:, 1]
            fold_time = time.time() - fold_start

            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_fold_val, proba)
            fold_scores.append(auc)
            fold_times.append(fold_time)

        # Build result row
        result = params.copy()
        result['mean_test_score'] = np.mean(fold_scores)
        result['std_test_score'] = np.std(fold_scores)
        result['mean_fit_time'] = np.mean(fold_times)
        result['std_fit_time'] = np.std(fold_times)
        for i, score in enumerate(fold_scores):
            result[f'fold_{i+1}_auc'] = score

        results.append(result)
        print(f"    Mean ROC-AUC: {result['mean_test_score']:.4f} "
                f"(+/- {result['std_test_score']:.4f})")


    results_df = pd.DataFrame(results)
    best_idx = results_df['mean_test_score'].idxmax()
    best_params = {k: results_df.loc[best_idx, k] for k in param_names}

    print(f"Best parameters: {best_params}")
    print(f"Best CV ROC-AUC: {results_df.loc[best_idx, 'mean_test_score']:.4f}")

    # Train final model on full training data with best parameters
    final_clf = OnlineDebiasMClassifier(
        batch_str=best_params['batch_str'],
        learning_rate=best_params['learning_rate'],
        min_epochs=best_params['min_epochs'],
        l2_strength=best_params['l2_strength'],
        w_l2=best_params['w_l2'],
        random_state=SEED,
    )

    X_train_final = X_train_prepared
    X_test_final = X_test_prepared

    start_time = time.time()
    final_clf.fit(X_train_final, y_train)
    predictions = final_clf.predict_proba(X_test_final)[:, 1]
    time_elapsed = time.time() - start_time

    # Save outputs
    full_output_dir = f'{partial_output_dir}/{run_type}/'
    os.makedirs(full_output_dir, exist_ok=True)

    predictions_df = pd.DataFrame({
        'sample_id': test_sample_ids,
        'prediction': predictions,
    })
    predictions_df.to_csv(f'{full_output_dir}/predictions.csv', index=False)

    with open(f'{full_output_dir}/train_time.txt', 'w') as f:
        f.write(str(time_elapsed))

    results_df.to_csv(f'{full_output_dir}/params_search.csv', index=False)

    return full_output_dir


def build_param_grid(run_type, train_batches):
    if run_type not in {'default', 'optimized'}:
        raise ValueError(f"Invalid run_type: {run_type}")

    if len(train_batches) == 1:
        batch_str_values = [0.0]
    else:
        batch_str_values = ['infer']

    if run_type == 'default':
        param_grid = {k: list(v) for k, v in BASE_DEFAULT_PARAMS.items()}
    else:
        param_grid = {k: list(v) for k, v in BASE_PARAM_GRIDS.items()}

    param_grid['batch_str'] = batch_str_values
    return param_grid


def list_param_combinations(param_grid):
    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    param_combinations = [
        dict(zip(param_names, v))
        for v in product(*param_values)
    ]
    return param_names, param_combinations


def config_output_dir(partial_output_dir, run_type, config_index):
    return os.path.join(
        partial_output_dir,
        run_type,
        'configs',
        f'config_{config_index:04d}',
    )


def run_single_config(X_train, y_train, X_test, test_sample_ids, train_subject_ids,
                      param_grid, run_type, partial_output_dir, config_index):
    param_names, param_combinations = list_param_combinations(param_grid)

    if config_index < 0 or config_index >= len(param_combinations):
        raise IndexError(
            f"Config index {config_index} out of range (max {len(param_combinations) - 1})."
        )

    params = param_combinations[config_index].copy()
    print(f"Running config {config_index} with params {params}.")

    X_train_scaled, X_test_scaled = apply_feature_selection(X_train, X_test)
    X_train_scaled = add_epsilon_to_zero_sum_rows(X_train_scaled)
    X_test_scaled = add_epsilon_to_zero_sum_rows(X_test_scaled)

    grouped_cv = build_stratified_group_cv(n_splits=5, random_state=SEED)
    fold_scores = []
    fold_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        grouped_cv.split(X_train_scaled, y_train, train_subject_ids)
    ):
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train_scaled[val_idx]
        y_fold_val = y_train[val_idx]

        clf = OnlineDebiasMClassifier(
            batch_str=params['batch_str'],
            learning_rate=params['learning_rate'],
            min_epochs=params['min_epochs'],
            l2_strength=params['l2_strength'],
            w_l2=params['w_l2'],
            random_state=SEED,
        )

        fold_start = time.time()
        clf.fit(X_fold_train, y_fold_train)
        proba = clf.predict_proba(X_fold_val)[:, 1]
        fold_time = time.time() - fold_start

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_fold_val, proba)
        fold_scores.append(auc)
        fold_times.append(fold_time)

    result = params.copy()
    result['mean_test_score'] = float(np.mean(fold_scores))
    result['std_test_score'] = float(np.std(fold_scores))
    result['mean_fit_time'] = float(np.mean(fold_times))
    result['std_fit_time'] = float(np.std(fold_times))
    for i, score in enumerate(fold_scores):
        result[f'fold_{i+1}_auc'] = float(score)

    output_dir = config_output_dir(partial_output_dir, run_type, config_index)
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([result]).to_csv(
        os.path.join(output_dir, 'params_search.csv'),
        index=False,
    )

    return output_dir


def aggregate_configs(X_train, y_train, X_test, test_sample_ids,
                      param_grid, run_type, partial_output_dir):
    _, param_combinations = list_param_combinations(param_grid)
    missing = []
    results = []

    for config_index in range(len(param_combinations)):
        output_dir = config_output_dir(partial_output_dir, run_type, config_index)
        params_path = os.path.join(output_dir, 'params_search.csv')
        if not os.path.exists(params_path):
            missing.append(output_dir)
            continue
        results.append(pd.read_csv(params_path))

    if missing:
        missing_list = "\n".join(missing)
        raise SystemExit(f"Missing config outputs:\n{missing_list}")

    results_df = pd.concat(results, ignore_index=True)

    run_output_dir = os.path.join(partial_output_dir, run_type)
    os.makedirs(run_output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(run_output_dir, 'params_search.csv'), index=False)

    best_idx = results_df['mean_test_score'].idxmax()
    best_params = results_df.loc[best_idx].to_dict()

    print(f"Best parameters: {best_params}")
    print(f"Best CV ROC-AUC: {results_df.loc[best_idx, 'mean_test_score']:.4f}")

    final_clf = OnlineDebiasMClassifier(
        batch_str=best_params['batch_str'],
        learning_rate=best_params['learning_rate'],
        min_epochs=int(best_params['min_epochs']),
        l2_strength=best_params['l2_strength'],
        w_l2=best_params['w_l2'],
        random_state=SEED,
    )

    fs_train, fs_test = apply_feature_selection(X_train, X_test)
    X_train_final, X_test_final = fs_train, fs_test
    X_train_final = add_epsilon_to_zero_sum_rows(X_train_final)
    X_test_final = add_epsilon_to_zero_sum_rows(X_test_final)

    start_time = time.time()
    final_clf.fit(X_train_final, y_train)
    predictions = final_clf.predict_proba(X_test_final)[:, 1]
    time_elapsed = time.time() - start_time

    predictions_df = pd.DataFrame({
        'sample_id': test_sample_ids,
        'prediction': predictions,
    })
    predictions_df.to_csv(os.path.join(run_output_dir, 'predictions.csv'), index=False)

    with open(os.path.join(run_output_dir, 'train_time.txt'), 'w') as f:
        f.write(str(time_elapsed))

    return run_output_dir


def run_default(X_train, y_train, X_test, test_sample_ids, train_subject_ids, partial_output_dir):
    """Run DEBIAS-M with default parameters, using CV paradigm with single param set."""
    print("Running DEBIAS-M with default parameters...")
    return run_with_params(
        X_train, y_train, X_test, test_sample_ids, train_subject_ids,
        DEFAULT_PARAMS, 'default', partial_output_dir,
    )


def run_optimized(X_train, y_train, X_test, test_sample_ids, train_subject_ids, partial_output_dir):
    """Run DEBIAS-M with hyperparameter search."""
    print("Running DEBIAS-M with hyperparameter search...")
    return run_with_params(
        X_train, y_train, X_test, test_sample_ids, train_subject_ids,
        PARAM_GRIDS, 'optimized', partial_output_dir,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEBIAS-M benchmarking runner')
    parser.add_argument('task_name')
    parser.add_argument('benchmark_datasets_dir')
    parser.add_argument(
        '--mode',
        choices=['full', 'list-configs', 'run-config', 'aggregate'],
        default='full',
    )
    parser.add_argument('--run-type', choices=['default', 'optimized'])
    parser.add_argument('--config-index', type=int)
    args = parser.parse_args()

    if args.mode in {'list-configs', 'run-config', 'aggregate'} and not args.run_type:
        raise SystemExit('--run-type is required for list-configs, run-config, and aggregate modes.')

    if args.mode == 'run-config' and args.config_index is None:
        raise SystemExit('--config-index is required for run-config mode.')

    output_dir = f'benchmarking_outputs/{args.task_name}_debias_m'

    if args.mode == 'list-configs':
        with contextlib.redirect_stdout(sys.stderr):
            X_train, y_train, X_test, test_sample_ids, train_subject_ids = load_benchmark_data(
                args.benchmark_datasets_dir,
                args.task_name,
                get_batch_info=True,
                return_subject_ids=True,
            )
        train_batches = np.unique(X_train[:, 0].astype(int))
        param_grid = build_param_grid(args.run_type, train_batches)
        _, param_combinations = list_param_combinations(param_grid)
        for idx, params in enumerate(param_combinations):
            print(f"{idx}\t{json.dumps(params, sort_keys=True)}")
        sys.exit(0)

    print(f"Loading data for task: {args.task_name}")
    X_train, y_train, X_test, test_sample_ids, train_subject_ids = load_benchmark_data(
        args.benchmark_datasets_dir,
        args.task_name,
        get_batch_info=True,
        return_subject_ids=True,
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    train_batches = np.unique(X_train[:, 0].astype(int))
    print(f"Train batches: {train_batches}")
    print(f"Test batch ID: {int(X_test[0, 0])}")

    if len(train_batches) == 1:
        DEFAULT_PARAMS['batch_str'] = [0.0]
        PARAM_GRIDS['batch_str'] = [0.0]
    else:
        DEFAULT_PARAMS['batch_str'] = ['infer']
        PARAM_GRIDS['batch_str'] = ['infer']

    if args.mode in {'run-config', 'aggregate'}:
        param_grid = build_param_grid(args.run_type, train_batches)
        if args.mode == 'run-config':
            run_single_config(
                X_train,
                y_train,
                X_test,
                test_sample_ids,
                train_subject_ids,
                param_grid,
                args.run_type,
                output_dir,
                args.config_index,
            )
        else:
            aggregate_configs(
                X_train,
                y_train,
                X_test,
                test_sample_ids,
                param_grid,
                args.run_type,
                output_dir,
            )
        sys.exit(0)

    # Full mode: run both default and optimized in one process (legacy behavior)
    default_grid = build_param_grid('default', train_batches)
    optimized_grid = build_param_grid('optimized', train_batches)

    print("\n" + "=" * 60)
    print("DEFAULT RUN")
    print("=" * 60)
    run_with_params(
        X_train,
        y_train,
        X_test,
        test_sample_ids,
        train_subject_ids,
        default_grid,
        'default',
        output_dir,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZED RUN")
    print("=" * 60)
    run_with_params(
        X_train,
        y_train,
        X_test,
        test_sample_ids,
        train_subject_ids,
        optimized_grid,
        'optimized',
        output_dir,
    )

    print("\nDEBIAS-M benchmarking complete.")
