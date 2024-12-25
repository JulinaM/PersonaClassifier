import pandas as pd
import numpy as np
import torch.nn as nn
import re,os, glob, traceback, nltk, logging, sys
from datetime import datetime
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from utils.DataProcessor import FeatureSelection, PreProcessor
from utils.Visualization import generate_cm, generate_auroc
from utils.Models import train_val_dl_models

global timestamp
global ckpt 
global logging

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

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
        return output

class Dataset:
    def __init__(self, filepath, emb_model, targets, demo):
        logging.info(f'Processing  {filepath} Dataset.')
        df = pd.read_csv(filepath) 
        if demo: df = df[:demo]
        self.contextual_emb = PreProcessor.process_embeddings(df, emb_model) if emb_model else []
        self.X = df.drop(['Unnamed: 0', 'STATUS'] + targets, axis=1)
        self.Y = df[targets]
        logging.info(f'X Shape: {self.X.shape} , Y Shape: {self.Y.shape}  Robert Emb Shape: {self.contextual_emb.shape}')


class My_training_class:
    def __init__(self, model_list=None, emb_model=None, demo=True,traits=None, ):
        self.models = model_list if model_list else ['svm', 'lr', 'rf', 'xgb', 'bilstm', 'mlp']
        self.emb_model = emb_model
        self.traits = traits if traits else ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU'] 
        self.demo = demo
        self.all_outputs ={}
        for model in model_list:
            self.all_outputs[model] = {}
    
    def read_dataset(self, train_file, val_file):
        self.train_set = Dataset(train_file, self.emb_model, self.traits, self.demo)    
        self.val_set = Dataset(val_file, self.emb_model, self.traits, self.demo)   
        return self.train_set, self.val_set

    def select_features(self, target_col):
        X, y = self.train_set.X, self.train_set.Y[[target_col]] 
        return FeatureSelection.mutual_info_selection(X, y)
        
    def prepare_dataset(self, stat_df, emb_df, y_df):
        # logging.info(f'{stat_df.shape}, {y_df.shape}, {emb_df.shape}')
        scaler = StandardScaler() 
        stat_features_scaled = scaler.fit_transform(stat_df)

        X = np.concatenate([stat_features_scaled, emb_df], axis=1)
        y = np.array(y_df) 
        y = y.ravel()  
        logging.info(f'statistical embedding: {stat_features_scaled.shape}')
        logging.info(f'contextual embedding: {emb_df.shape} ')
        logging.info(f'total embedding X and y: {X.shape} and {y.shape}')
        logging.info(f'Data Preparation Completed.')
        return X, y

    def get_tensor(self, X, y, batch_size=16, shuffle=True):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def init_models(self, X_shape):
        for model in self.models:
            if model =='svm':
                self.svm_model = SVC(kernel='linear')
            elif model == 'lr':
                self.lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
            elif model == 'rf':
                self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model == 'xgb':
                self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
            elif model == 'bilstm':   
                self.bilstm_model = BiLSTMClassifier(input_dim= X_shape, hidden_dim=128, output_dim=1, num_layers=2, bidirectional=True, do_attention=True, dropout_rate=0.5)
            elif model == 'mlp':  
                self.mlp_model = MLP(input_size=X_shape, hidden_size=128, output_size=1, dropout_rate=0.3)
        logging.info(f'Model Initiated.')
    
    def fit_models(self, X_train, y_train, X_val, y_val, target_col):
        logging.info(f'Fitting and Validating Models...')
        for model in self.models:
            if model =='svm':
                self.svm_model.fit(X_train, y_train)
                svm_y_pred = self.svm_model.predict(X_val)
                svm_y_probs = self.svm_model.predict_proba(X_val)[:, 1]
                svm_accuracy = accuracy_score(y_val, svm_y_pred)
                logging.info(f'SVM Val Acc: {svm_accuracy:.2f}')
                self.all_outputs[model][target_col] = (y_val, svm_y_pred, svm_y_probs)
            elif model == 'lr':
                self.lr_model.fit(X_train, y_train)
                lr_y_pred = self.lr_model.predict(X_val)
                lr_y_probs = self.lr_model.predict_proba(X_val)[:, 1]
                lr_accuracy = accuracy_score(y_val, lr_y_pred)
                self.all_outputs[model][target_col] = (y_val, lr_y_pred, lr_y_probs)
                logging.info(f'LR Val Acc: {lr_accuracy:.2f}')
            elif model == 'rf':
                self.rf_model.fit(X_train, y_train)
                rf_y_pred = self.rf_model.predict(X_val)
                rf_y_proba = self.rf_model.predict_proba(X_val)[:, 1]
                rf_accuracy = accuracy_score(y_val, rf_y_pred)
                self.all_outputs[model][target_col] = (y_val, rf_y_pred, rf_y_proba)
                logging.info(f'RF Val Acc: {rf_accuracy:.2f}')
            elif model == 'xgb': 
                self.xgb_model.fit(X_train, y_train)
                # self.xgb_model.save_model(f"{ckpt}/{self.xgb_model.__class__.__name__}_{target_col}.json")
                xgb_y_pred = self.xgb_model.predict(X_val)
                xgb_y_proba = self.xgb_model.predict_proba(X_val)[:, 1]
                xgb_accuracy = accuracy_score(y_val, xgb_y_pred)
                self.all_outputs[model][target_col] = (y_val, xgb_y_pred, xgb_y_proba)
                logging.info(f'SGBoost Val Acc: {xgb_accuracy:.2f}')
            elif model == 'bilstm':   
                train_loader = self.get_tensor(X_train, y_train)
                val_loader = self.get_tensor(X_val, y_val)
                self.bilstm_acc, y_pred, y_scores = train_val_dl_models(self.bilstm_model, train_loader, val_loader)
                self.all_outputs[model][target_col] = (y_val, y_pred, y_scores)
                torch.save(self.bilstm_model.state_dict(), f"{ckpt}/{self.bilstm_model.__class__.__name__}_{target_col}.pth")
            elif model == 'mlp':  
                train_loader = self.get_tensor(X_train, y_train)
                val_loader = self.get_tensor(X_val, y_val)  
                self.mlp_acc, y_pred, y_scores = train_val_dl_models(self.mlp_model, train_loader, val_loader)
                self.all_outputs[model][target_col] = (y_val, y_pred, y_scores)
                torch.save(self.mlp_model.state_dict(), f"{ckpt}/{self.mlp_model.__class__.__name__}_{target_col}.pth")

    def display_metrics(self, savefig=True):
        performance_records = {} 
        for model in self.models:
            logging.info(15*'='+f" {model} "+ 15*'=')
            a_output = self.all_outputs[model]
            results = performance_records = generate_cm(a_output, model, ckpt, True)
            generate_auroc(a_output, model, ckpt, True)
        # performance_df = pd.DataFrame(performance_records)
        # logging.info(f"Performance metrics dataframe created with shape: {performance_df.shape}")
        # performance_df.to_csv(f"{ckpt}/performance.csv")
        # # for col in self.traits:
        #     s = performance_df[performance_df['Classifier'] ==col]
        #     best_model_row = s.loc[s['Accuracy'].idxmax()]
        #     logging.info(f'For {best_model_row["Classifier"]}, {best_model_row["Model"]},  {best_model_row["Accuracy"]}')

if __name__ == "__main__":
    try:
        emb = sys.argv[1]
        models = sys.argv[2]
        print(emb, models)
        emb_models = {'1':'roberta-base', '2':'bert-base-uncased', '3':'vinai/bertweet-base', '4':'xlnet-base-cased'}
        emb = emb_models[emb] if emb in emb_models.keys() else None
        models = ['lr', 'rf', 'xgb', 'mlp', 'bilstm'] if models == 'all' else ["mlp"]
        print(emb, models)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ckpt = f"checkpoint/{emb.split('-')[0]}-{timestamp}" if emb else f"checkpoint/{timestamp}"
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        logging.basicConfig(filename=f'{ckpt}/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       
        my_train = My_training_class(model_list=models, emb_model=emb, demo=None)
        my_train.read_dataset('./processed_data/pandora_train.csv', './processed_data/pandora_val.csv')
        logging.info(30*"*")
        for target_col in my_train.traits:
            logging.info(10*"-")
            logging.info(target_col)
            logging.info(10*"-")
            selected_features = my_train.select_features(target_col)
            logging.info(f'Selected Features for {target_col} : {selected_features}')
            X_train, y_train = my_train.prepare_dataset(my_train.train_set.X[selected_features],
                my_train.train_set.contextual_emb, 
                my_train.train_set.Y[[target_col]])
            X_val, y_val = my_train.prepare_dataset(my_train.val_set.X[selected_features], 
                my_train.val_set.contextual_emb, 
                my_train.val_set.Y[[target_col]])
            my_train.init_models(X_shape=X_train.shape[1])
            my_train.fit_models(X_train, y_train, X_val, y_val, target_col)
            logging.info(30*"*")
        my_train.display_metrics()
        
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
