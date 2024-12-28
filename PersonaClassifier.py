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
from utils.Models import train_val_dl_models, train_with_kfold_val_dl_models, MLP, evaluate_on_test_dataset
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
global timestamp
global ckpt 
global logging

hyperparameters = {
    'kfold' : False,
    'hidden_dim' : 128,
    'dropout_rate' : 0.3,
    'batch_size': 16,
    'epochs': 16,
    'learning_rate': 0.001,
    'random_state': 42,
    'max_grad_norm': 1
}

class Dataset:
    def __init__(self, filepath, emb_model, targets, demo):
        logging.info(f'Processing  {filepath} Dataset.')
        df = pd.read_csv(filepath) 
        if demo: df = df.sample(demo)
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

    def select_features(self, X, y):
        return FeatureSelection.mutual_info_selection(X, y) #TODO experiment with other feature selection
        
    def prepare_dataset(self, stat_df, emb_df, y_df):
        # logging.info(f'{stat_df.shape}, {y_df.shape}, {emb_df.shape}')
        scaler = StandardScaler() 
        stat_features_scaled = scaler.fit_transform(stat_df) #TODO experiment with other tranformation like log
        X = np.concatenate([stat_features_scaled, emb_df], axis=1)
        y = np.array(y_df) 
        y = y.ravel()  
        logging.info(f'statistical embedding: {stat_features_scaled.shape}')
        logging.info(f'contextual embedding: {emb_df.shape} ')
        logging.info(f'total embedding X and y: {X.shape} and {y.shape}')
        logging.info(f'Data Preparation Completed.')
        return X, y

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
                self.bilstm_model = BiLSTMClassifier(input_dim=X_shape, hidden_dim=128, output_dim=1, num_layers=2, bidirectional=True, do_attention=True, dropout_rate=0.5)
            elif model == 'mlp':  
                self.mlp_model = MLP(input_size=X_shape, hidden_size=128, output_size=1, dropout_rate=0.3)
        logging.info(f'Model Initiated.')
    
    def fit_models(self, X_train, y_train, X_val, y_val, target_col, save_ckpt=False):
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
                if save_ckpt: self.xgb_model.save_model(f"{ckpt}/{self.xgb_model.__class__.__name__}_{target_col}.json")
            elif model == 'bilstm':   
                bilstm_acc, y_pred, y_probas = train_val_dl_models(self.bilstm_model, X_train, y_train, X_val, y_val)
                self.all_outputs[model][target_col] = (y_val, y_pred, y_probas)
                logging.info(f'BILSTM Val Acc: {bilstm_acc:.2f}')
                if save_ckpt: torch.save(self.bilstm_model.state_dict(), f"{ckpt}/{self.bilstm_model.__class__.__name__}_{target_col}.pth")
            elif model == 'mlp':  
                mlp_acc, y_pred, y_probas = train_val_dl_models(self.mlp_model, X_train, y_train, X_val, y_val)
                self.all_outputs[model][target_col] = (y_val, y_pred, y_probas)
                logging.info(f'MLP Val Acc: {mlp_acc:.2f}')
                if save_ckpt: torch.save(self.mlp_model.state_dict(), f"{ckpt}/{self.mlp_model.__class__.__name__}_{target_col}.pth")
        logging.info(f'Model Fitted.')

    def display_metrics(self, all_outputs, Test=False, savefig=True):
        logging.info(f'Generating Metrics and Figures.')
        performance_records = {} 
        for model in all_outputs:
            logging.info(15*'='+f" {model} "+ 15*'=')
            a_output = all_outputs[model]
            (cm, auroc, perf) = ('cm_test', 'auroc_test', 'performance_test') if Test else ('cm', 'auroc', 'performance')
            performance_records[model] = generate_cm(a_output, f'{ckpt}/{model}_{cm}.png')
            generate_auroc(a_output, model, f'{ckpt}/{model}_{auroc}.png')
            performance_df = pd.DataFrame(performance_records)
            logging.info(f"Performance df shape: {performance_df.shape}")
            performance_df.to_csv(f"{ckpt}/{perf}.csv")
            # # for col in self.traits:
            #     s = performance_df[performance_df['Classifier'] ==col]
            #     best_model_row = s.loc[s['Accuracy'].idxmax()]
            #     logging.info(f'For {best_model_row["Classifier"]}, {best_model_row["Model"]},  {best_model_row["Accuracy"]}')

def kfold_train(emb, models, demo):
    logging.info(f'K-Fold Training started: {emb} {models} {demo}')
    my_train = My_training(model_list=models, emb_model=emb, demo=demo)
    dataset = Dataset('./processed_data/2-splits/pandora_train_val.csv', my_train.emb_model, my_train.traits, my_train.demo)    
    test_dataset = Dataset('./processed_data/2-splits/pandora_test.csv', my_train.emb_model, my_train.traits, my_train.demo)    

    logging.info(50*"*")
    train_outtputs, test_outputs = {'mlp':{}}, {'mlp':{}}
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        selected_features = my_train.select_features(dataset.X, dataset.Y[[target_col]])
        logging.info(f'Selected Features for {target_col} : {selected_features}')
        X, y = my_train.prepare_dataset(dataset.X[selected_features], dataset.contextual_emb, dataset.Y[[target_col]])
        mlp_model = MLP(input_size=X.shape[1], hidden_size=128, output_size=1, dropout_rate=0.3)
        mlp_acc, y_pred, y_probas, y_val = train_with_kfold_val_dl_models(mlp_model, X, y)
        logging.info(f"Val Accuracy: {target_col}: {mlp_acc}")
        train_outtputs['mlp'][target_col] = (y_val, y_pred, y_probas)
        #Evaluate
        X_test, y_test = my_train.prepare_dataset(test_dataset.X[selected_features], test_dataset.contextual_emb, test_dataset.Y[[target_col]])
        acc, preds, probas = evaluate_on_test_dataset(mlp_model, X_test, y_test)
        test_outputs['mlp'][target_col] = (y_test, preds, probas)
        logging.info(f"Test Accuracy: {target_col}: {acc}")
    my_train.display_metrics(train_outtputs)
    my_train.display_metrics(test_outputs, True)
   
def train(emb, models, demo):
    logging.info(f'Training started: {emb} {models} {demo}')
    my_train = My_training(model_list=models, emb_model=emb, demo=demo)
    train_set = Dataset('./processed_data/3-splits/pandora_train.csv', my_train.emb_model, my_train.traits, my_train.demo)   
    val_set = Dataset('./processed_data/3-splits/pandora_val.csv', my_train.emb_model, my_train.traits, my_train.demo)    
    logging.info(50*"*")
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        selected_features = my_train.select_features(train_set.X, train_set.Y[[target_col]] )
        logging.info(f'Selected Features for {target_col} : {selected_features}')
        X_train, y_train = my_train.prepare_dataset(train_set.X[selected_features], train_set.contextual_emb, train_set.Y[[target_col]])
        X_val, y_val = my_train.prepare_dataset(val_set.X[selected_features], val_set.contextual_emb, val_set.Y[[target_col]])
        my_train.init_models(X_shape=X_train.shape[1])
        my_train.fit_models(X_train, y_train, X_val, y_val, target_col, save_ckpt=False)
        logging.info(50*"-")
    my_train.display_metrics(my_train.all_outputs)

    
if __name__ == "__main__":
    try:
        emb = sys.argv[1]
        models = sys.argv[2]
        kfold = True #sys.argv[3]
        demo = None
        print(emb, models, kfold, demo)
        emb_models = {'1':'roberta-base', '2':'bert-base-uncased', '3':'vinai/bertweet-base', '4':'xlnet-base-cased'}
        emb = emb_models[emb] if emb in emb_models.keys() else None
        models = ['lr', 'rf', 'xgb', 'mlp', 'bilstm'] if models == 'all' else [ 'mlp']
        print(emb, models, kfold, demo)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ckpt = f"checkpoint/{emb.split('-')[0]}-{timestamp}" if emb else f"checkpoint/{timestamp}"
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        logging.basicConfig(filename=f'{ckpt}/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       
        if kfold: kfold_train(emb, models, demo)
        else: train(emb, models, demo)

    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
