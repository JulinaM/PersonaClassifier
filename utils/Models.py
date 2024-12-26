import pandas as pd
import numpy as np
import re,os, glob, traceback, nltk, logging, sys
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid =  nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
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

def train_val_dl_models(model, train_loader, val_loader, max_grad_norm=1.0, epochs=16, lr=0.001):
    logging.info(f'{model.__class__.__name__}; lr={lr}')
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    history = {}
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for t, labels in train_loader:
            optimizer.zero_grad()  
            outputs= model(t)
            loss = criterion(outputs.squeeze(), labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        model.eval()  
        val_labels, val_probas, val_loss = [], [], 0
        with torch.no_grad():  
            for inputs, labels in val_loader:
                outputs= model(inputs)
                val_loss += criterion(outputs.squeeze(), labels).item()  
                val_probas.append(outputs)  
                val_labels.append(labels) 
        val_probas = torch.cat(val_probas)
        val_labels = torch.cat(val_labels)
        val_preds = (val_probas > 0.5).float() 
        val_accuracy = accuracy_score(val_labels.numpy(), val_preds.numpy())
        if epoch % 4 == 0:
            # history['val_acc'] = val_accuracy
            # history['val_loss'] = val_loss
            # # history['train_acc'] = val_accuracy
            # history['train_loss'] = total_loss
            logging.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, 'f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')
    return val_accuracy, val_preds, val_probas

def k_fold_train_val_dl_models(model, dataset, k=5, batch_size=32, max_grad_norm=1.0, epochs=16, lr=0.001):
    """
    Perform k-fold cross-validation for a deep learning model.
    
    Parameters:
    - model_class: A callable to instantiate the model (e.g., a class name).
    - dataset: A PyTorch Dataset containing the data.
    - k: Number of folds.
    - batch_size: Batch size for data loaders.
    - max_grad_norm: Maximum gradient norm for clipping.
    - epochs: Number of training epochs.
    - lr: Learning rate.
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    logging.info(f"Performing {k}-Fold Cross-Validation")
    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        logging.info(f"Fold {fold + 1}/{k}")
        # Split dataset into training and validation
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size)

        val_accuracy, val_preds, val_probas = train_single_fold(model, train_loader=train_loader, val_loader=val_loader, max_grad_norm=max_grad_norm, epochs=epochs, lr=lr)
        fold_results.append(val_accuracy)
        logging.info(f"Fold {fold + 1} Accuracy: {val_accuracy:.4f}")

    avg_accuracy = sum(fold_results) / k
    logging.info(f"Average Accuracy across {k} folds: {avg_accuracy:.4f}")
    return fold_results, avg_accuracy
