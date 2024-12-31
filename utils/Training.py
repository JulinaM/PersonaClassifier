import pandas as pd
import numpy as np
import re,os, glob, traceback, nltk, logging, sys
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset, Subset

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_one_epoch(model, train_loader, criterion, optimizer, max_grad_norm, threshold=0.5):
    model.train()
    correct, train_loss = 0, 0.0
    for input, target in train_loader:
        optimizer.zero_grad()  
        output = model(input)
        loss = criterion(output, target.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        train_loss += loss.item()
        probs = torch.sigmoid(output)
        pred = (probs > threshold).float() 
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /=len(train_loader)
    train_accuracy =  correct / len(train_loader.dataset)
    return train_accuracy, train_loss

def validate_one_epoch(model, val_loader, criterion, threshold=0.5):
    correct, val_loss = 0,  0.0
    val_preds, val_probas, val_targets = [], [], []
    model.eval()  
    with torch.no_grad():  
        for input, target in val_loader:
            output = model(input)
            if criterion: val_loss += criterion(output, target.unsqueeze(1)).item()  
            probs = torch.sigmoid(output) 
            pred = (probs > threshold).float() 
            correct += pred.eq(target.view_as(pred)).sum().item()
            val_probas.append(probs)  
            val_preds.append(pred)
            val_targets.append(target)
    if criterion: val_loss /= len(val_loader)
    val_accuracy = correct / len(val_loader.dataset)
    return val_accuracy, val_loss, val_preds, val_probas, val_targets

def make_prediction(model, X, threshold=0.5):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()  
    with torch.no_grad():  
        output = model(X_tensor)
        probs = torch.sigmoid(output) 
        pred = (probs > threshold).float() 
    return pred.numpy(), probs.numpy()

def evaluate_on_test_dataset(model, X, y, batch_size=32, threshold=0.5):
    logging.info(f'{model.__class__.__name__}; threshold={threshold}, batch_size={batch_size}')
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_acc, _, test_preds, test_probas, _ = validate_one_epoch(model, test_loader, criterion=None, threshold=threshold)
    return test_acc, torch.cat(test_preds), torch.cat(test_probas)

def train_val_dl_models(model, X_train, y_train, X_val, y_val, ckpt, batch_size=32, epochs=32, lr=0.001, max_grad_norm=1.0):
    logging.info(f'{model.__class__.__name__}; lr={lr}, batch_size={batch_size}')
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()  # criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_accuracies, val_accuracies = [], []
    early_stopper = EarlyStopper(patience=3, min_delta=0)
    for epoch in range(epochs):
        train_accuracy, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, max_grad_norm)
        train_accuracies.append(train_accuracy)

        val_accuracy, val_loss, val_preds, val_probas, _ = validate_one_epoch(model, val_loader, criterion)
        val_accuracies.append(val_accuracy)
        if epoch % 4 == 0:
            logging.info(f'Epoch: [{epoch + 1}/{epochs}], Train:: Loss: {train_loss:.4f}, Acc:{train_accuracy:.4f}, and  Val:: Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f} ')
        if early_stopper.early_stop(val_loss):             
            break
    # display_acc_curve(train_accuracies, val_accuracies, epoch, ckpt)
    return val_accuracy, torch.cat(val_preds), torch.cat(val_probas)

def display_acc_curve(train_accuracies, val_accuracies, num_epochs, ckpt):
    import matplotlib.pyplot as plt
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(f'{ckpt}/acc_curve.png', dpi=300, bbox_inches='tight') 
    plt.show()

def train_with_kfold_val_dl_models(model, X, y, k_folds=5, batch_size=32, epochs=16, lr=0.001, max_grad_norm=1.0):
    logging.info(f'{model.__class__.__name__}; lr={lr}, batch_size={batch_size}, k_folds={k_folds}')
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = {}
    early_stopper = EarlyStopper(patience=3, min_delta=0)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()  
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            train_accuracy, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, max_grad_norm)

        val_accuracy, val_loss, val_preds, val_probas, val_targets = validate_one_epoch(model, val_loader,  criterion)
        fold_results[fold] = {'train_loss': train_loss, 'train_acc': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy}
        logging.info(f'Fold {fold+1}/{k_folds} - Train:: Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} and Val:: Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}')
        if early_stopper.early_stop(val_loss):   
            logging.info(f'For {fold+1}, Early stopping at epoch {epoch+1}' )
            break
    # logging.info(fold_results)
    return val_accuracy, torch.cat(val_preds), torch.cat(val_probas), torch.cat(val_targets)



