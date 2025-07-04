import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

import os
import sys


# Define a simple fully connected neural network
class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # assuming binary classification
        )

    def forward(self, x):
        return self.model(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, ff_hidden_dim=64, num_classes=2):
        """
        Transformer-based classifier for tabular data without positional encoding.
        """
        super(TransformerClassifier, self).__init__()

        # Each scalar feature gets projected to a d_model-dimensional vector
        self.embedding = nn.Linear(1, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_hidden_dim,
            batch_first=False  # PyTorch Transformer expects (seq_len, batch, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, input_dim)
        # Step 1: Treat each scalar feature as a token => shape becomes (batch_size, input_dim, 1)
        x = x.unsqueeze(-1)

        # Step 2: Embed each token (feature) => (batch_size, input_dim, d_model)
        x = self.embedding(x)

        # Step 3: Transpose to (input_dim, batch_size, d_model) as required by Transformer
        x = x.transpose(0, 1)

        # Step 4: Transformer encoder
        x = self.transformer(x)

        # Step 5: Aggregate token outputs by mean pooling over the sequence (input_dim dimension)
        x = x.mean(dim=0)  # Shape: (batch_size, d_model)

        # Step 6: Final classification
        logits = self.classifier(x)  # Shape: (batch_size, num_classes)
        return logits


def prepare_data(train, test_x):
    # Prepare data
    drop_cols = ['label', 'sample_id', 'subject_id']
    X_train = train.drop(columns=drop_cols).values.astype('float32')
    y_train = train['label'].values.astype('int64')
    X_test = test_x.drop(columns=['sample_id', 'subject_id']).values.astype('float32')
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    return X_train_tensor, y_train_tensor, X_test_tensor


def train_and_predict_proba(train_loader, X_test_tensor, input_dim, model_type):
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'transformer':
        model = TransformerClassifier(input_dim=input_dim).to(device)
    elif model_type == 'fully_connected':
        model = FCNN(input_dim=input_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    model.train()
    for epoch in range(20):  # number of epochs
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    # Prediction
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities


if __name__ == '__main__':
    task_name, benchmark_datasets_dir, model_name = sys.argv[1:]
    train_path = f"{benchmark_datasets_dir}/{task_name}_train.csv"
    test_x_path = f"{benchmark_datasets_dir}/{task_name}_test.csv"
    print('cuda available: ', torch.cuda.is_available())

    train = pd.read_csv(train_path)
    test_x = pd.read_csv(test_x_path)

    X_train_tensor, y_train_tensor, X_test_tensor = prepare_data(train, test_x)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    probabilities = train_and_predict_proba(train_loader, X_test_tensor, input_dim=X_train_tensor.shape[1],
                                            model_type=model_name)

    model_dir = f'{model_name}_default'
    os.makedirs(model_dir, exist_ok=True)
    pd.DataFrame(index=test_x.sample_id, data=probabilities)[[1]].to_csv(
        f"{model_dir}/{task_name}_predictions_on_test.csv")
