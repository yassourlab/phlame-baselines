import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import pandas as pd
import sys
import itertools
import os
import time
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from benchmark_utils import load_benchmark_data


# ---------------------------
# Seed for Reproducibility
# ---------------------------
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Models
# ---------------------------
class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Single output neuron
        )

    def forward(self, x):
        return self.model(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, ff_hidden_dim=64):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)  # Single output neuron

    def forward(self, x):
        x = x.unsqueeze(-1)             # (batch, features, 1)
        x = self.embedding(x)           # (batch, features, d_model)
        x = self.transformer(x)         # (batch, features, d_model)
        x = x.mean(dim=1)               # mean pooling over features
        return self.classifier(x)


# ---------------------------
# Data prep
# ---------------------------
def prepare_data(train, test_x):
    # train already has sample_id and subject_id removed by load_benchmark_data
    X_train = train.drop(columns=['label']).values.astype('float32')
    y_train = train['label'].values.astype('float32')  # Changed to float32 for BCEWithLogitsLoss
    # test_x still has sample_id and subject_id
    X_test = test_x.drop(columns=['sample_id']).values.astype('float32')
    return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test)


# ---------------------------
# Training for one split
# ---------------------------
def train_model(train_loader, val_loader, input_dim, model_type, params, device):
    if model_type == 'transformer':
        model = TransformerClassifier(
            input_dim=input_dim,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            ff_hidden_dim=params["ff_hidden_dim"]
        ).to(device)
    elif model_type == 'fully_connected':
        model = FCNN(
            input_dim=input_dim,
            hidden_dim=params["hidden_dim"]
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    for epoch in range(params["epochs"]):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch).squeeze()  # Squeeze to match y_batch shape
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # Validation AUC
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            y_pred.extend(probs)
            y_true.extend(y_batch.numpy())

    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5  # if only one class present in val set

    return auc


# ---------------------------
# Cross-validation evaluation
# ---------------------------
def cross_val_score_auc(X, y, input_dim, model_type, params, device, folds=5):
    """
    Perform cross-validation and return detailed metrics.
    
    Returns:
        dict with keys: 'fold_scores', 'fold_times', 'mean_score', 'std_score', 'mean_time', 'std_time'
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    dataset = TensorDataset(X, y)

    scores = []
    times = []
    for train_idx, val_idx in kf.split(X):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=params["batch_size"], shuffle=False)

        start_time = time.time()
        auc = train_model(train_loader, val_loader, input_dim, model_type, params, device)
        fold_time = time.time() - start_time
        
        scores.append(auc)
        times.append(fold_time)

    import numpy as np
    return {
        'fold_scores': scores,
        'fold_times': times,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_time': np.mean(times),
        'std_time': np.std(times)
    }


def predict_proba(model, X_test_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor.to(device)).squeeze()
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return probabilities


# ---------------------------
# Main
# ---------------------------

DEFAULT_PARAMS = {
    "fully_connected": {
        "hidden_dim": 64,
        "lr": 1e-3,
        "epochs": 20,
        "batch_size": 128,
    },
    "transformer": {
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2,
        "ff_hidden_dim": 64,
        "lr": 1e-3,
        "epochs": 20,
        "batch_size": 32,
    }
}

PARAM_GRIDS = {
    "fully_connected": {
        "hidden_dim": [16, 32, 64, 128],
        "lr": [1e-2, 1e-3, 1e-4],
        "epochs": [20, 50, 100],
        "batch_size": [64, 128, 256],
    },
    "transformer": {
        "d_model": [16, 32],
        "nhead": [2, 4],
        "num_layers": [1, 2, 3],
        "ff_hidden_dim": [32, 64],
        "lr": [1e-3, 1e-4],
        "epochs": [10, 20],
        "batch_size": [64],
    }
}


def run_with_params(train, test_x, model_name, param_combinations, run_type, device, task_name):
    """
    Run model with given parameter combinations using CV paradigm.
    
    Args:
        train: Training data
        test_x: Test features
        model_name: Name of the model to run ('transformer' or 'fully_connected')
        param_combinations: List of parameter dictionaries to evaluate
        run_type: String identifier for output directory ('default' or 'optimized')
        device: PyTorch device
        task_name: Task name for output directory
    
    Returns:
        full_output_dir: Path to the output directory
    """
    X_train_tensor, y_train_tensor, X_test_tensor = prepare_data(train, test_x)
    
    # Setup output directory
    os.makedirs('benchmarking_outputs', exist_ok=True)
    output_dir = f'benchmarking_outputs/{task_name}_{model_name}'
    full_output_dir = f'{output_dir}/{run_type}/'
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Track all parameter search results
    search_results = []
    best_auc, best_params = 0, None
    
    print(f"Running hyperparameter search with {len(param_combinations)} combinations...")
    
    for i, params in enumerate(param_combinations):
        print(f"Evaluating combination {i+1}/{len(param_combinations)}: {params}")
        cv_results = cross_val_score_auc(
            X_train_tensor, y_train_tensor, 
            X_train_tensor.shape[1],
            model_name, params, device, folds=5
        )
        
        # Store results for this parameter combination
        result_row = params.copy()
        result_row['mean_test_score'] = cv_results['mean_score']
        result_row['std_test_score'] = cv_results['std_score']
        result_row['mean_fit_time'] = cv_results['mean_time']
        result_row['std_fit_time'] = cv_results['std_time']
        
        # Add individual fold scores
        for j, score in enumerate(cv_results['fold_scores']):
            result_row[f'fold_{j+1}_auc'] = score
        
        search_results.append(result_row)
        
        # Track best parameters
        if cv_results['mean_score'] > best_auc:
            best_auc = cv_results['mean_score']
            best_params = params
    
    # Train final model with best params on full training data
    if model_name == "transformer":
        best_model = TransformerClassifier(
            input_dim=X_train_tensor.shape[1],
            d_model=best_params["d_model"],
            nhead=best_params["nhead"],
            num_layers=best_params["num_layers"],
            ff_hidden_dim=best_params["ff_hidden_dim"]
        ).to(device)
    else:
        best_model = FCNN(
            input_dim=X_train_tensor.shape[1],
            hidden_dim=best_params["hidden_dim"]
        ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["lr"])
    full_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                             batch_size=best_params["batch_size"], shuffle=True)
    
    start_time = time.time()
    for epoch in range(best_params["epochs"]):
        best_model.train()
        for X_batch, y_batch in full_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = best_model(X_batch).squeeze()
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    
    time_elapsed = time.time() - start_time
    
    # Get predictions
    best_model.eval()
    with torch.no_grad():
        logits = best_model(X_test_tensor.to(device)).squeeze()
        predictions = torch.sigmoid(logits).cpu().numpy()
    
    # Save predictions
    pd.DataFrame({
        'sample_id': test_x['sample_id'],
        'prediction': predictions
    }).to_csv(f'{full_output_dir}/predictions.csv', index=False)
    
    # Save train time
    with open(f'{full_output_dir}/train_time.txt', 'w') as f:
        f.write(str(time_elapsed))
    
    # Save parameter search results
    params_search_df = pd.DataFrame(search_results)
    params_search_df.to_csv(f'{full_output_dir}/params_search.csv', index=False)
    
    return full_output_dir


def run_default(train, test_x, model_name, device, task_name):
    """Run model with default parameters, using CV paradigm with single parameter set."""
    params = DEFAULT_PARAMS[model_name]
    # Create single-element list for parameter combinations
    param_combinations = [params]
    
    return run_with_params(train, test_x, model_name, param_combinations, 'default', device, task_name)


def run_optimized(train, test_x, model_name, device, task_name):
    """Run model with hyperparameter search, logging all iterations."""
    param_grid = PARAM_GRIDS[model_name]
    
    # Generate all parameter combinations from grid
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    return run_with_params(train, test_x, model_name, param_combinations, 'optimized', device, task_name)


if __name__ == '__main__':
    # Set seed for reproducibility
    set_seed(42)
    
    task_name, benchmark_datasets_dir, model_name = sys.argv[1:]
    train, test_x = load_benchmark_data(benchmark_datasets_dir, task_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda available: ', torch.cuda.is_available())
    
    # Default run
    print(f"Starting {model_name} default run")
    default_output_dir = run_default(train, test_x, model_name, device, task_name)
    
    # Optimized run with hyperparameter search
    print(f"Starting {model_name} hyperparameter search")
    optimized_output_dir = run_optimized(train, test_x, model_name, device, task_name)
