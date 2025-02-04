import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt


# Sample Neural Network Model for Regression
class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super(RegressionModel, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Single output for regression
        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for regression
        return x

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
        loss = criterion(outputs, targets.unsqueeze(1))  # Ensure target shape matches output
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping
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
    logging.info(f'Training {model.__class__.__name__} with batch_size={batch_size}, dropout={model.dropout}')
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()  # Use MSELoss for RMSE calculation
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

