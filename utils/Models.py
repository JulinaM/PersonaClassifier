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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.sigmoid =  nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # x = self.sigmoid(x)
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional=True, do_attention=True, dropout_rate=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.do_attention = do_attention
        self.attention = DotProductAttention(hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim) 
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)  
        self.dropout = nn.Dropout(dropout_rate)  
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
        output = self.sigmoid(output)
        return output

# # Test model Initialize model
# model = BiLSTMClassifier(input_dim=768, hidden_dim=128, output_dim=5, num_layers=2)
# input_data = torch.randn(2, 5, 768)  # Example input (batch_size=32, seq_len=50, input_dim=768)
# output= model(input_data)
# print(output.shape)  # Expected output: (batch_size, output_dim)

def evaluate_on_test_dataset(model, X, y, batch_size=32, threshold=0.5):
    logging.info(f'{model.__class__.__name__}; threshold={threshold}, batch_size={batch_size}')
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_preds, test_probas, correct = [], [], 0
    model.eval()  
    with torch.no_grad():  
        for input, target in test_loader:
            output = model(input)
            probs = torch.sigmoid(output) 
            pred = (probs > threshold).float() 
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_probas.append(probs)  
            test_preds.append(pred)
    test_acc = correct / len(test_loader.dataset)
    return test_acc, torch.cat(test_preds), torch.cat(test_probas)

def train_val_dl_models(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=16, lr=0.001, max_grad_norm=1.0):
    logging.info(f'{model.__class__.__name__}; lr={lr}, batch_size={batch_size}')
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()  # criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
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
            pred = (probs > 0.5).float() 
            correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /=len(train_loader)
        train_accuracy =  correct / len(train_loader.dataset)

        val_preds, val_probas, val_targets = [], [], []
        correct, val_loss = 0,  0.0
        model.eval()  
        with torch.no_grad():  
            for input, target in val_loader:
                output = model(input)
                val_loss += criterion(output, target.unsqueeze(1)).item()  
                probs = torch.sigmoid(output) 
                pred = (probs > 0.5).float() 
                correct += pred.eq(target.view_as(pred)).sum().item()
                val_probas.append(probs)  
                val_preds.append(pred)
                val_targets.append(target)
        val_loss /= len(val_loader)
        val_accuracy = correct / len(val_loader.dataset)
        val_acc = accuracy_score(torch.cat(val_targets).numpy(), torch.cat(val_preds).numpy())
        if epoch % 4 == 0:
            logging.info(f'Epoch: [{epoch + 1}/{epochs}], Train:: Loss: {train_loss:.4f}, Acc:{train_accuracy:.4f}, and  Val:: Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f} ')
    return val_accuracy, torch.cat(val_preds), torch.cat(val_probas)

def train_with_kfold_val_dl_models(model, X, y, k_folds=5, batch_size=32, epochs=16, lr=0.001, max_grad_norm=1.0):
    logging.info(f'{model.__class__.__name__}; lr={lr}, batch_size={batch_size}, k_folds={k_folds}')

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()  
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
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
                pred = (probs > 0.5).float() 
                correct += pred.eq(target.view_as(pred)).sum().item()
            train_loss /=len(train_loader)
            train_accuracy =  correct / len(train_loader.dataset)

        val_preds, val_probas, val_targets = [], [], []
        correct, val_loss = 0,  0.0
        model.eval()  
        with torch.no_grad():  
            for input, target in val_loader:
                output = model(input)
                val_loss += criterion(output, target.unsqueeze(1)).item()  
                probs = torch.sigmoid(output) 
                pred = (probs > 0.5).float() 
                correct += pred.eq(target.view_as(pred)).sum().item()
                val_probas.append(probs)  
                val_preds.append(pred)
                val_targets.append(target)
        val_loss /= len(val_loader)
        val_accuracy = correct / len(val_loader.dataset)
        val_acc = accuracy_score(torch.cat(val_targets).numpy(), torch.cat(val_preds).numpy())
        fold_results[fold] = {'train_loss': train_loss, 'train_acc': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy}
        logging.info(f'Fold {fold+1}/{k_folds} - Train:: Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} and Val:: Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}')
    # logging.info(fold_results)
    return val_accuracy, torch.cat(val_preds), torch.cat(val_probas), torch.cat(val_targets)

