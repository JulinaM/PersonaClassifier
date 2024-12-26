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
from utils.Models import train_val_dl_models, MLP
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
global timestamp
global ckpt 
global logging

class Dataset:
    def __init__(self, filepath, emb_model, targets, demo):
        logging.info(f'Processing  {filepath} Dataset.')
        df = pd.read_csv(filepath) 
        if demo: df = df[:demo]
        self.contextual_emb = PreProcessor.process_embeddings(df, emb_model) if emb_model else []
        self.X = df.drop(['Unnamed: 0', 'STATUS'] + targets, axis=1)
        self.Y = df[targets]
        logging.info(f'X Shape: {self.X.shape}, Y Shape: {self.Y.shape},  Contextual Emb Shape: {self.contextual_emb.shape}')

class My_training:
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
                xgb_y_pred = self.xgb_model.predict(X_val)
                xgb_y_proba = self.xgb_model.predict_proba(X_val)[:, 1]
                xgb_accuracy = accuracy_score(y_val, xgb_y_pred)
                self.all_outputs[model][target_col] = (y_val, xgb_y_pred, xgb_y_proba)
                logging.info(f'SGBoost Val Acc: {xgb_accuracy:.2f}')
                # self.xgb_model.save_model(f"{ckpt}/{self.xgb_model.__class__.__name__}_{target_col}.json")
            elif model == 'bilstm':   
                train_loader = self.get_tensor(X_train, y_train)
                val_loader = self.get_tensor(X_val, y_val)
                bilstm_acc, y_pred, y_probas = train_val_dl_models(self.bilstm_model, train_loader, val_loader)
                self.all_outputs[model][target_col] = (y_val, y_pred, y_probas)
                logging.info(f'BILSTM Val Acc: {bilstm_acc:.2f}')
                torch.save(self.bilstm_model.state_dict(), f"{ckpt}/{self.bilstm_model.__class__.__name__}_{target_col}.pth")
            elif model == 'mlp':  
                train_loader = self.get_tensor(X_train, y_train)
                val_loader = self.get_tensor(X_val, y_val)  
                mlp_acc, y_pred, y_probas = train_val_dl_models(self.mlp_model, train_loader, val_loader)
                self.all_outputs[model][target_col] = (y_val, y_pred, y_probas)
                logging.info(f'MLP Val Acc: {mlp_acc:.2f}')
                # torch.save(self.mlp_model.state_dict(), f"{ckpt}/{self.mlp_model.__class__.__name__}_{target_col}.pth")
        logging.info(f'Model Fitted.')

    def display_metrics(self, savefig=True):
        logging.info(f'Generating Metrics and Figures.')
        performance_records = {} 
        for model in self.models:
            logging.info(15*'='+f" {model} "+ 15*'=')
            a_output = self.all_outputs[model]
            performance_records[model] = generate_cm(a_output, model, ckpt, True)
            generate_auroc(a_output, model, ckpt, True)
        performance_df = pd.DataFrame(performance_records)
        logging.info(f"Performance metrics shape: {performance_df.shape}")
        performance_df.to_csv(f"{ckpt}/performance.csv")
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
        models = ['lr', 'rf', 'xgb', 'mlp', 'bilstm'] if models == 'all' else ["xgb"]
        print(emb, models)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ckpt = f"checkpoint/{emb.split('-')[0]}-{timestamp}" if emb else f"checkpoint/{timestamp}"
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        logging.basicConfig(filename=f'{ckpt}/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       
        logging.info(f'Training started: {emb} {models} ')
        my_train = My_training(model_list=models, emb_model=emb, demo=None)
        my_train.read_dataset('./processed_data/pandora_train.csv', './processed_data/pandora_val.csv')
        logging.info(30*"*")
        for target_col in my_train.traits:
            logging.info(10*"-")
            logging.info(target_col)
            logging.info(10*"-")
            selected_features = my_train.select_features(target_col)
            logging.info(f'Selected Features for {target_col} : {selected_features}')
            X_train, y_train = my_train.prepare_dataset(my_train.train_set.X[selected_features], my_train.train_set.contextual_emb, my_train.train_set.Y[[target_col]])
            X_val, y_val = my_train.prepare_dataset(my_train.val_set.X[selected_features], my_train.val_set.contextual_emb, my_train.val_set.Y[[target_col]])
            my_train.init_models(X_shape=X_train.shape[1])
            my_train.fit_models(X_train, y_train, X_val, y_val, target_col)
            logging.info(30*"*")
        my_train.display_metrics()
        
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
