from dm_benchmark import DeepMicrobiome
import pandas as pd
import os
import sys
import time
import itertools
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import random
import tensorflow as tf
sys.path.append('..')
from benchmark_utils import load_benchmark_data
# # original hyperparameters
# AE_HYPER_PARAMETERS = {'dims':[[32],[64],[128],[256],[512],
#                         [64,32],[128,64],[256,128],[512,256],
#                         [128,64,32],[256,128,64],[512,256,128]]}

# VAE_HYPER_PARAMETERS = {'latent_dim': [4, 8, 16],'hidden_units': [4, 8,16,32,64]}

# CAE_HYPER_PARAMETERS = {'conv_layers': [2, 3], 'n_filter_on_first_layer': [4,8,16,32,64],}

# rf_hyper_parameters = [{'n_estimators': [s for s in range(100, 1001, 200)],
#                         'max_features': ['sqrt', 'log2'],
#                         'min_samples_leaf': [1, 2, 3, 4, 5],
#                         'criterion': ['gini', 'entropy']
#                         }, ]
# #svm_hyper_parameters_pasolli = [{'C': [2 ** s for s in range(-5, 16, 2)], 'kernel': ['linear']},
# #                        {'C': [2 ** s for s in range(-5, 16, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],
# #                         'kernel': ['rbf']}]
# svm_hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
#                         {'C': [2 ** s for s in range(-5, 6, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],'kernel': ['rbf']}]
# mlp_hyper_parameters = [{'numHiddenLayers': [1, 2, 3],
#                             'epochs': [30, 50, 100, 200, 300],
#                             'numUnits': [10, 30, 50, 100],
#                             'dropout_rate': [0.1, 0.3],
#                             },]
N_SPLITS = 5
NA_PARAMS_PLACEHOLDER = 'NA'
# benchmark hyperparameters
AE_HYPER_PARAMETERS = {
    'dims': [[64], [128], [256], [64, 32], [128, 64], [256, 128]]
}

VAE_HYPER_PARAMETERS = {
    'dims': [[8], [16], [32], [16, 8], [32, 16], [64, 32]]
}

CAE_HYPER_PARAMETERS = {
    'dims': [[32], [16], [32,16,8],[16,8,4],[16,8],[8,4]]
}

RF_HYPER_PARAMETERS = {
    'criterion': ['gini', 'entropy']
    
    
}

SVM_HYPER_PARAMETERS = {
    'kernel': ['linear', 'rbf']
    
    
}

MLP_HYPER_PARAMETERS = {
    'numHiddenLayers': [1, 2]

}

# Seed for reproducibility (consistent with other benchmark scripts)
SEED = 17


def set_seeds(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_data(benchmark_datasets_dir, task_name):
    """Load train and test datasets using shared benchmark_utils function."""
    train_df, X_test_df = load_benchmark_data(benchmark_datasets_dir, task_name)
    
    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    X_test = X_test_df.drop(columns=['sample_id']).values
    
    # Load ground truth for y_test
    test_y_path = f"{benchmark_datasets_dir}/{task_name}_test_gt.csv"
    y_test = pd.read_csv(test_y_path)['label'].values
    
    return X_train, y_train, X_test, y_test, X_test_df


def setup_output_directory(task_name, subdir):
    """Create and return output directory path."""
    output_dir = f'../benchmarking_outputs/{task_name}_deep_micro/{subdir}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def train_autoencoder(dm, ae_type, ae_params):
    """Train autoencoder based on type and parameters."""
    if 'dims' in ae_params:    
        dims = ae_params['dims']    
        if ae_type == 'ae':
            dm.ae(dims=dims, verbose=0)
        elif ae_type == 'vae':
            dm.vae(dims=dims, verbose=0)
        elif ae_type == 'cae':
            dm.cae(dims=dims, verbose=0)
    else:
        dm.cae(verbose=0)

def train_classifier(dm, clf_type, clf_params, cv=None):
    """Train classifier based on type and parameters. cv=None means no cross-validation."""
    if clf_type == 'rf':
        hyper_params = [clf_params] if clf_params else [{}]
        return dm.classification(hyper_parameters=hyper_params, method='rf',
                                cv=cv, n_jobs=-1, scoring='roc_auc')
    elif clf_type == 'svm':
        hyper_params = [clf_params] if clf_params else [{}]
        return dm.classification(hyper_parameters=hyper_params, method='svm',
                                cv=cv, n_jobs=-1, scoring='roc_auc')
    elif clf_type == 'mlp':
        hyper_params = clf_params if clf_params else {}
        return dm.classification(hyper_parameters=hyper_params, method='mlp',
                                cv=cv, n_jobs=-1, scoring='roc_auc')


def save_results(full_output_dir, X_test_df, predictions, time_elapsed):
    """Save predictions and training time."""
    pd.DataFrame({
        'sample_id': X_test_df['sample_id'],
        'prediction': predictions
    }).to_csv(f'{full_output_dir}/predictions.csv', index=False)
    
    with open(f'{full_output_dir}/train_time.txt', 'w') as f:
        f.write(str(time_elapsed))


def run_with_configs(task_name, benchmark_datasets_dir, ae_configs, clf_configs, run_type):
    """
    Run DeepMicro with given autoencoder and classifier configurations using CV paradigm.
    
    Args:
        task_name: Task name
        benchmark_datasets_dir: Directory containing benchmark datasets
        ae_configs: List of tuples (ae_type, ae_hyperparameters_dict)
        clf_configs: List of tuples (clf_type, clf_hyperparameters_dict)
        run_type: String identifier for output directory ('default' or 'optimized')
    
    Returns:
        full_output_dir: Path to the output directory
    """
    X_train, y_train, X_test, y_test, X_test_df = load_data(benchmark_datasets_dir, task_name)
    full_output_dir = setup_output_directory(task_name, run_type)
    
    # Generate all autoencoder parameter combinations
    all_ae_configs = []
    for ae_type, ae_params in ae_configs:
        ae_param_names = list(ae_params.keys())
        ae_param_values = list(ae_params.values())
        ae_param_combinations = list(itertools.product(*ae_param_values))
        for ae_param_combo in ae_param_combinations:
            ae_param_dict = dict(zip(ae_param_names, ae_param_combo))
            all_ae_configs.append((ae_type, ae_param_dict))
    
    # Generate all classifier parameter combinations
    all_clf_configs = []
    for clf_type, clf_params in clf_configs:
        clf_param_names = list(clf_params.keys())
        clf_param_values = list(clf_params.values())
        clf_param_combinations = list(itertools.product(*clf_param_values))
        for clf_param_combo in clf_param_combinations:
            clf_param_dict = dict(zip(clf_param_names, clf_param_combo))
            all_clf_configs.append((clf_type, clf_param_dict))
    
    print(f"Running hyperparameter search for DeepMicro...")
    print(f"  {len(all_ae_configs)} autoencoder configs × {len(all_clf_configs)} classifier configs × {N_SPLITS} folds")
    print(f"  = {len(all_ae_configs) * len(all_clf_configs) * N_SPLITS} total evaluations")
    print(f"  Training {len(all_ae_configs) * N_SPLITS} autoencoders (optimized structure)")
    
    # Collect all fold results
    all_fold_results = []
    
    # Setup KFold splits once
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    # OUTER LOOP: Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n=== Processing Fold {fold_idx + 1}/{N_SPLITS} ===")
        
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Create temporary output directory for this fold
        temp_dir = f'{full_output_dir}/temp_fold_{fold_idx}'
        os.makedirs(temp_dir, exist_ok=True)
        
        # MIDDLE LOOP: Iterate through autoencoder configurations
        for ae_type, ae_param_dict in all_ae_configs:
            print(f" Training Autoencoder: {ae_type} with params {ae_param_dict}")
            ae_start_time = time.time()
            
            # Initialize DeepMicrobiome with fold data
            dm = DeepMicrobiome(X_train_fold, X_val_fold, y_train_fold, y_val_fold, 
                              task_name, temp_dir, seed=SEED)
            
            # Train autoencoder once for this fold and ae configuration
            train_autoencoder(dm, ae_type, ae_param_dict)
            ae_train_time = time.time() - ae_start_time
            
            # INNER LOOP: Iterate through classifier configurations
            for clf_type, clf_param_dict in all_clf_configs:
                clf_start_time = time.time()
                
                # Train classifier on the encoded representations
                val_prob = train_classifier(dm, clf_type, clf_param_dict)
                
                # Calculate AUC score
                fold_auc = roc_auc_score(y_val_fold, val_prob[:, 1])
                clf_train_time = time.time() - clf_start_time
                
                # Store this fold result
                result = {
                    'fold_idx': fold_idx,
                    'autoencoder': ae_type,
                    'classifier': clf_type,
                    **{f'ae_{k}': str(v) for k, v in ae_param_dict.items()},
                    **{f'clf_{k}': str(v) for k, v in clf_param_dict.items()},
                    'auc_score': fold_auc,
                    'ae_fit_time': ae_train_time,
                    'clf_fit_time': clf_train_time,
                    'total_fit_time': ae_train_time + clf_train_time
                }
                all_fold_results.append(result)
        
        # Clean up temporary directory for this fold
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Convert to DataFrame for easy aggregation
    results_df = pd.DataFrame(all_fold_results)
    
    # Aggregate results by configuration (across folds)
    # Group by all config columns (exclude fold_idx, scores, and times)
    config_cols = [col for col in results_df.columns 
                   if col not in ['fold_idx', 'auc_score', 'ae_fit_time', 'clf_fit_time', 'total_fit_time']]
    
    aggregated = results_df.fillna(NA_PARAMS_PLACEHOLDER).groupby(config_cols).agg({
        'auc_score': ['mean', 'std'],
        'ae_fit_time': 'mean',
        'clf_fit_time': 'mean',
        'total_fit_time': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in aggregated.columns.values]
    
    # Rename for clarity
    aggregated.rename(columns={
        'auc_score_mean': 'mean_test_score',
        'auc_score_std': 'std_test_score',
        'ae_fit_time_mean': 'mean_ae_fit_time',
        'clf_fit_time_mean': 'mean_clf_fit_time',
        'total_fit_time_mean': 'mean_fit_time',
        'total_fit_time_std': 'std_fit_time'
    }, inplace=True)
    
    # Add individual fold scores to aggregated results
    for fold_idx in range(N_SPLITS):
        fold_scores = results_df[results_df['fold_idx'] == fold_idx].set_index(config_cols)['auc_score']
        aggregated[f'fold_{fold_idx + 1}_auc'] = aggregated.set_index(config_cols).index.map(fold_scores).values
    
    # Find best configuration
    best_idx = aggregated['mean_test_score'].idxmax()
    best_row = aggregated.iloc[best_idx]
    
    best_config = {
        'ae_type': best_row['autoencoder'],
        'ae_params': {k.replace('ae_', ''): best_row[k] for k in best_row.index if k.startswith('ae_')},
        'clf_type': best_row['classifier'],
        'clf_params': {k.replace('clf_', ''): best_row[k] for k in best_row.index if k.startswith('clf_')}
    }
    
    # Convert string representations back to proper types for ae_params
    if 'dims' in best_config['ae_params']:
        import ast
        best_config['ae_params']['dims'] = ast.literal_eval(best_config['ae_params']['dims'])
    
    print(f"\n=== Hyperparameter Search Complete ===")
    print(f"Best configuration: {best_config['ae_type']} + {best_config['clf_type']}")
    print(f"  Mean AUC: {best_row['mean_test_score']:.4f} ± {best_row['std_test_score']:.4f}")
    
    # Train final model with best configuration on full training data
    print("\nTraining final model with best config on full training data...")
    
    start_time = time.time()
    dm_final = DeepMicrobiome(X_train, X_test, y_train, y_test, task_name, full_output_dir, seed=SEED)
    train_autoencoder(dm_final, best_config['ae_type'], best_config['ae_params'])
    test_prob = train_classifier(dm_final, best_config['clf_type'], 
                                 {k:v for k,v in best_config['clf_params'].items() if v!=NA_PARAMS_PLACEHOLDER}, 
                                 cv=None)
    time_elapsed = time.time() - start_time
    
    # Save results
    save_results(full_output_dir, X_test_df, test_prob[:, 1], time_elapsed)
    
    # Save parameter search results
    aggregated.to_csv(f'{full_output_dir}/params_search.csv', index=False)
    results_df.to_csv(f'{full_output_dir}/all_fold_results.csv', index=False)
    
    return full_output_dir


def run_default(task_name, benchmark_datasets_dir):
    """Run DeepMicro with default parameters, using CV paradigm with single configuration."""
    # Default configuration: CAE autoencoder with RF classifier, no hyperparameters
    ae_configs = [('cae', {})]
    clf_configs = [('rf', {})]
    
    return run_with_configs(task_name, benchmark_datasets_dir, ae_configs, clf_configs, 'default')


def run_optimized(task_name, benchmark_datasets_dir):
    """Run DeepMicro with hyperparameter search, testing all autoencoder and classifier combinations."""
    # Define autoencoder types and their hyperparameters
    ae_configs = [
        ('cae', CAE_HYPER_PARAMETERS),
        ('ae', AE_HYPER_PARAMETERS),
        ('vae', VAE_HYPER_PARAMETERS),
    ]
    
    # Define classifier types and their hyperparameters
    clf_configs = [
        ('rf', RF_HYPER_PARAMETERS),
        ('svm', SVM_HYPER_PARAMETERS),
        ('mlp', MLP_HYPER_PARAMETERS)
    ]
    
    return run_with_configs(task_name, benchmark_datasets_dir, ae_configs, clf_configs, 'optimized')


if __name__ == '__main__':
    # Set seeds once for reproducibility
    set_seeds(SEED)
    
    # parse args
    task_name, benchmark_datasets_dir = sys.argv[1:]
    
    
    # Default run
    print("Starting DeepMicro default run")
    default_output_dir = run_default(task_name, benchmark_datasets_dir)
    
    # # Optimized run with hyperparameter search
    print("Starting DeepMicro hyperparameter search")
    optimized_output_dir = run_optimized(task_name, benchmark_datasets_dir)
