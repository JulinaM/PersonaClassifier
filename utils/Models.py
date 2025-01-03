import torch.nn as nn
import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from utils.Training import train_val_kfold, train_val, predict
import logging

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.sigmoid =  nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.dropout(x)
        # x = self.fc3(x)
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
        # output = self.sigmoid(output)
        return output

# # Test model Initialize model
# model = BiLSTMClassifier(input_dim=768, hidden_dim=128, output_dim=5, num_layers=2)
# input_data = torch.randn(2, 5, 768)  # Example input (batch_size=32, seq_len=50, input_dim=768)
# output= model(input_data)
# print(output.shape)  # Expected output: (batch_size, output_dim)

class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, kFold=5, epochs=32, batch_size=16, lr=0.001, device=None):
        self.model = model
        # self.optimizer_class = optimizer_class
        # self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.kFold = kFold
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Unique class labels
        if self.kFold:
            return train_val_kfold(self.model, X, y, self.kFold, self.batch_size, self.epochs, self.lr)
        else:   
            return train_val(self.model, X, y, self.batch_size, self.epochs, self.lr)

    def predict(self, X):
        pred, _ = predict(self.model, X)
        return pred
        # return self.classes_[pred]  
    
    def predict_proba(self, X):
        _, probas = predict(self.model, X)
        return probas

# from skorch import NeuralNetClassifier
# import torch.nn as nn
# import torch
# mlp = MLP(input_size=X.shape[1], hidden_size=128, output_size=1, dropout_rate=0.5)
# skorchMLP = NeuralNetClassifier(
#     mlp,
#     criterion=nn.BCEWithLogitsLoss,
#     optimizer=torch.optim.Adam,
#     lr=0.001,
#     max_epochs=32,
# )

class IdentityEstimator(BaseEstimator, ClassifierMixin):
    '''
    An identity estimator used for calibrating probability data
    '''
    def __init__(self):
        self.classes_= [0, 1]
        # pass
 
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state
 
    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
 
    def fit(self, X, y, sample_weight=None):
        return self
 
    def predict_proba(self, X):
        assert X.shape[1] == 1
        probs = np.concatenate((1 - X, X), axis=1)
        return probs
