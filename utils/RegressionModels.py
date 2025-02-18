import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Single output for regression
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  
        x = torch.sigmoid(x) * 100  
        return x

class DotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
    def forward(self, x):
        query = x[:, -1:, :]  # Shape: (batch_size, 1, hidden_dim * 2)
        scores = torch.bmm(query, x.transpose(1, 2))  # Shape: (batch_size, 1, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # Shape: (batch_size, seq_len, 1)
        context_vector = torch.bmm(attention_weights, x)  # Shape: (batch_size, 1, hidden_dim * 2)
        return context_vector, attention_weights

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_layers=2, bidirectional=True, do_attention=True, output_dim=1):
        super(BiLSTMClassifier, self).__init__()
        self.do_attention = do_attention
        self.attention = DotProductAttention(hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim) 
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)  
        self.dropout = nn.Dropout(dropout)  
        self.sigmoid =  nn.Sigmoid()
    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)  
        if self.do_attention:
            context_vector, attention_weights = self.attention(x)
            context_vector = self.layer_norm1(context_vector)
        else:
            context_vector = x
        lstm_output, _ = self.lstm(context_vector)     
        lstm_output = self.layer_norm2(lstm_output)
        last_hidden_state = lstm_output[:, -1, :]  # Shape: (batch_size, hidden_dim * 2)
        last_hidden_state = self.dropout(last_hidden_state)
        output = self.fc(last_hidden_state)  
        output = self.sigmoid(output) *100
        return output

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def _train_one_epoch(model, train_loader, device, criterion, optimizer, max_grad_norm):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # No sigmoid for regression
        loss = criterion(outputs, targets.unsqueeze(1))  
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)  # Compute mean loss
    # train_rmse = train_loss ** 0.5  # Convert MSE to RMSE
    return train_loss

def _validate_one_epoch(model, val_loader, device, criterion):
    val_loss = 0.0
    val_preds, val_targets = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            val_loss += loss.item()
            val_preds.append(outputs)
            val_targets.append(targets)
    val_loss /= len(val_loader)
    return val_loss, torch.cat(val_preds), torch.cat(val_targets)

def plot_learning_curve(train_losses, val_losses, filepath):
    """Plots train vs validation loss over epochs"""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curve")
    plt.legend()
    if filepath: plt.savefig(filepath)
    plt.show()

def train_val(model, X_train, y_train, X_val, y_val, device, batch_size, epochs, optimizer, max_grad_norm=1.0):
    logging.info(f'Training {model.__class__.__name__} with batch_size={batch_size}, dropout={model.dropout}, epochs={epochs}')
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()  # Use MSELoss for RMSE calculation
    # criterion = nn.MAELoss()
    early_stopper = EarlyStopper(patience=3, min_delta=0.001)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss = _train_one_epoch(model, train_loader, device, criterion, optimizer, max_grad_norm)
        val_loss, val_preds, val_targets = _validate_one_epoch(model, val_loader, device, criterion)
        if epoch % 4 == 0:
            logging.info(f'Epoch [{epoch + 1}/{epochs}], Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}')
        if early_stopper.early_stop(val_loss):          
            logging.info(f'Early stopping at epoch {epoch}.')   
            break
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return train_losses, val_losses

def train(model, X_train, y_train, device, batch_size, epochs, optimizer, max_grad_norm=1.0):
    logging.info(f'Training {model.__class__.__name__} with batch_size={batch_size}, dropout={model.dropout}, epochs={epochs}')
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = nn.MSELoss()  # Use MSELoss for RMSE calculation
    # criterion = nn.MAELoss()
    train_losses =  []
    for epoch in range(epochs):
        train_loss = _train_one_epoch(model, train_loader, device, criterion, optimizer, max_grad_norm)
        if epoch % 4 == 0:
            logging.info(f'Epoch [{epoch + 1}/{epochs}], Train MSE: {train_loss:.4f}')
        train_losses.append(train_loss)
    return train_losses
    
def evaluate_model(model, X_test, y_test, device, tolerance=0.10):
    def regression_accuracy(preds, targets, tolerance=0.10):
        abs_error = np.abs(preds - targets)
        within_tolerance = abs_error <= (tolerance * np.abs(targets))  # Check if error is within 10%
        accuracy = np.mean(within_tolerance) * 100 
        return accuracy

    def mean_absolute_percentage_error(preds, targets):
        return np.mean(np.abs((targets - preds) / targets)) * 100  

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval() 
    preds, targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Get model predictions
            preds.append(outputs.cpu())  # Store predictions
            targets.append(labels.cpu())  # Store true values
    preds = torch.cat(preds).numpy().flatten()  # Flatten predictions
    targets = torch.cat(targets).numpy().flatten()  # Flatten actual values
    # Compute evaluation metrics
    mse = mean_squared_error(targets, preds)  # RMSE , squared=False
    mae = mean_absolute_error(targets, preds)  # MAE
    r2 = r2_score(targets, preds)  # R² Score
    acc = regression_accuracy(preds, targets, tolerance)
    mape = mean_absolute_percentage_error(preds, targets)
    logging.info(f"Test MSE: {mse:.4f}")
    logging.info(f"Test MAE: {mae:.4f}")
    logging.info(f"Test R² Score: {r2:.4f}")
    logging.info(f"Test Accuracy (Within {tolerance*100}% Tolerance): {acc:.2f}%")
    logging.info(f"Test MAPE: {mape:.2f}%")
    return preds, targets, mse, mae, r2, acc, mape

def train_val_kfold(model, X, y, k_folds, device, batch_size, epochs, optimizer, max_grad_norm=1.0):
    logging.info(f'{model.__class__.__name__};  batch_size={batch_size}, k_folds={k_folds}, epochs={epochs}')
    k_folds = 5
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    targets = np.array([target for _, target in dataset]) 
    # kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = {}
    criterion = nn.MSELoss()  # Use MSELoss for RMSE calculation

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, targets)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            train_loss = _train_one_epoch(model, train_loader, device, criterion, optimizer, max_grad_norm)
        val_loss, val_preds, val_targets = _validate_one_epoch(model, val_loader, device, criterion)
        fold_results[fold] = {'train_loss': train_loss,  'val_loss': val_loss}
        logging.info(f'Fold {fold+1}/{k_folds} - Train:: Loss: {train_loss:.4f} and Val:: Loss: {val_loss:.4f}')
    avg_val_loss = sum(fold['val_loss'] for fold in fold_results.values()) / k_folds
    logging.info(f'Average Val loss: {avg_val_loss}')
    return avg_val_loss