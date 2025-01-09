import pandas as pd
import numpy as np
import torch.nn as nn
import re,os, glob, traceback, nltk, logging, sys
from datetime import datetime
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from utils.DataProcessor import FeatureSelection, PreProcessor
from utils.Visualization import generate_cm, generate_auroc, display_auroc, display_calibration, calculate_threshold, generate_cal_result
from utils.Models import MLP, MLPWrapper, BiLSTMClassifier, IdentityEstimator
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from utils.Training import _train_one_epoch, _validate_one_epoch, predict, EarlyStopper
from torch.utils.data import DataLoader, TensorDataset, Subset
global ckpt 
global logging

class ModelCreator:
    def __init__(self, X_shape, kFold, hyperparameters):
        bilstm = BiLSTMClassifier(input_dim=X_shape, hidden_dim=hyperparameters['hidden_dim'], output_dim=1, num_layers=2, bidirectional=True, do_attention=True, dropout_rate=hyperparameters["dropout_rate"])
        mlp = MLP(input_size=X_shape, hidden_size=hyperparameters['hidden_dim'], output_size=1, dropout_rate=hyperparameters["dropout_rate"])
        self.estimators = {
            "svm" : SVC(kernel='linear'),
            "lr" : LogisticRegression(solver='lbfgs', max_iter=1000),
            "rf" : RandomForestClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
            'bilstm': MLPWrapper(model=bilstm, kFold=kFold, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'], lr=hyperparameters['learning_rate']),
            'mlp': MLPWrapper(model=mlp, kFold=kFold, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'], lr=hyperparameters['learning_rate'])
        }

class Dataset:
    def __init__(self, filepath, emb_model, targets, demo):
        logging.info(f'Processing  {filepath} Dataset.')
        NRC_VAD = ['Valence', 'Arousal', 'Dominance']
        NRC_emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'sent_score']
        SENTIMENT = ['sent_score']
        df = pd.read_csv(filepath) 
        if demo: df = df.sample(demo)
        self.contextual_emb = PreProcessor.process_embeddings(df, emb_model) if emb_model else []
        self.X = df.drop(['Unnamed: 0', 'STATUS'] + targets, axis=1)
        self.Y = df[targets]
        self.ORIGINAL = df[['STATUS'] + targets]
        logging.info(f'X Shape: {self.X.shape}, Y Shape: {self.Y.shape},  Contextual Emb Shape: {self.contextual_emb.shape if emb_model else []}')

class My_training:
    def __init__(self, model_list=None, emb_model=None, demo=True, traits=None):
        self.models = model_list if model_list else ['svm', 'lr', 'rf', 'xgb', 'bilstm', 'mlp']
        self.emb_model = emb_model
        self.traits = traits if traits else ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU'] 
        self.demo = demo
        self.test_outputs ={}
        self.val_outputs = {}
        self.cal_outputs ={}
        self.estimators = {}
        for model in model_list:
            self.test_outputs[model] = {}
            self.val_outputs[model]= {}
            self.cal_outputs[model] = {}
        self.test_df = pd.DataFrame()
        
    def prepare_dataset(self, stat_df, emb_df, y_df, features=None):
        # logging.info(f'{stat_df.shape}, {y_df.shape}, {emb_df.shape}')
        scaler = StandardScaler() #TODO experiment with other tranformation like log
        X_df = pd.DataFrame(scaler.fit_transform(stat_df), columns=stat_df.columns)
        if features is None: features = FeatureSelection.filter_selection(X_df, y_df, 5)
        X = np.concatenate([X_df[features], emb_df], axis=1)
        y = np.array(y_df).ravel()
        logging.info(f'statistical embedding: {X_df[features].shape}')
        logging.info(f'contextual embedding: {emb_df.shape} ')
        logging.info(f'Final shape X and y: {X.shape} and {y.shape}')
        logging.info(f'Data Preparation Completed.')
        return X, y, features

    def init_models(self, X_shape, kFold, hyperparameters):
        model_creator = ModelCreator(X_shape, kFold, hyperparameters)
        for model in self.models:
            self.estimators[model] = model_creator.estimators[model]
        logging.info(f'Model Initiated.')
    
    def fit_and_validate(self, X, y, target_col, save_ckpt=False):
        logging.info(f'Fitting and Validating Models...')
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
        for model in self.models:
            estimator = self.estimators[model]
            if model in ['mlp', 'bilstm']:
                self.estimators[model].set_val_data(X_val, y_val)
            estimator.fit(X_train, y_train)
            y_pred, y_prob = estimator.predict(X_val), estimator.predict_proba(X_val)[:,1]
            self.val_outputs[model][target_col] = (y_val, y_pred, y_prob)
            logging.info(f'{model} VAl Acc: {accuracy_score(y_val, y_pred):.2f}')
        logging.info(f'Model Fitted and Validated.')

    def fit(self, X, y, target_col, save_ckpt=False):
        logging.info(f'Fitting Models...')
        for model in self.models:
            estimator = self.estimators[model]
            estimator.fit(X, y)
            if save_ckpt: estimator.save_model(f"{ckpt}/{estimator.__class__.__name__}_{target_col}.json")
        logging.info(f'Model Fitted.')

    def evaluate_models(self, X, y, target_col, calibrate=False):
        logging.info(f'Evaluating on Test Dataset.')
        for model in self.models:
            estimator = self.estimators[model]
            pred = estimator.predict(X)
            probs = estimator.predict_proba(X)[:, 1] #handle for svm decision_function
            if calibrate: generate_cal_result(y, pred, probs, target_col, f'{ckpt}/calibration/uncal_{model}_{target_col}.png')
            self.test_outputs[model][target_col] = (y, pred, probs)
            # self.test_df[f'{model}_{target_col}'] = pred
            logging.info(f'{model} Test Acc: {accuracy_score(y, pred):.2f}')
        logging.info(f'Evaluation completed.')

    def calibrate_models(self, X, y, X_test, y_test, target_col):
        # for model in self.models:
        model = "mlp"
        y_prob_val = self.mlpWrapper.predict_proba(X)
        y_prob_test = self.mlpWrapper.predict_proba(X_test)
        calibrated = CalibratedClassifierCV(base_estimator = IdentityEstimator(), method = 'sigmoid', cv = 5)
        calibrated.fit(y_prob_val, y)
        y_cal_pred, y_cal_prob = calibrated.predict(y_prob_test), calibrated.predict_proba(y_prob_test)[:, 1]
        generate_cal_result(y_test, y_cal_pred, y_cal_prob, target_col, f'{ckpt}/calibration/cal_{model}_{target_col}.png' )
        self.cal_outputs[model][target_col] = (y_test, y_cal_pred, y_cal_prob)    
        logging.info(f'Calibration completed.')

    def display_metrics(self, all_outputs, initial=None, savefig=True):
        logging.info(f'Generating Metrics and Figures.')
        performance_records = {} 
        for model in all_outputs:
            logging.info(15*'='+f" {model} "+ 15*'=')
            a_output = all_outputs[model]
            (cm, auroc, perf) = (f'cm_{initial}', f'auroc_{initial}', f'performance_{initial}') if initial else ('cm', 'auroc', 'performance')
            performance_records[model] = generate_cm(a_output, f'{ckpt}/{model}_{cm}.png')
            generate_auroc(a_output, model, f'{ckpt}/{model}_{auroc}.png')
            performance_df = pd.DataFrame(performance_records)
            logging.info(f"Performance df shape: {performance_df.shape}")
            if savefig: performance_df.to_csv(f"{ckpt}/{perf}.csv")
            # # for col in self.traits:
            #     s = performance_df[performance_df['Classifier'] ==col]
            #     best_model_row = s.loc[s['Accuracy'].idxmax()]
            #     logging.info(f'For {best_model_row["Classifier"]}, {best_model_row["Model"]},  {best_model_row["Accuracy"]}')

def kfold_train(emb, models, demo):
    logging.info(f'K-Fold Training started: {emb} {models} {demo}')
    my_train = My_training(model_list=models, emb_model=emb, demo=demo)
    dataset = Dataset('./processed_data/2-splits/pandora_train_val.csv', my_train.emb_model, my_train.traits, my_train.demo)    
    test_dataset = Dataset('./processed_data/2-splits/pandora_test.csv', my_train.emb_model, my_train.traits, my_train.demo)    
    my_train.test_df = test_dataset.ORIGINAL
    logging.info(50*"*")

    lr, epochs, batch_size, k_folds = 0.001, 32, 16, 5
    val_outputs, test_outputs, cal_test_outputss, selected_features = {'mlp':{}}, {'mlp':{}}, {'mlp':{}}, {}
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = {}
        X, y, fs = my_train.prepare_dataset(dataset.X, dataset.contextual_emb, dataset.Y[[target_col]], features=[])
        selected_features[target_col] = fs
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, y_train = Subset(X, train_idx), Subset(y, train_idx)
            X_val, y_val = Subset(X, val_idx), Subset(y, val_idx)

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
            train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = MLP(input_size=X.shape[1], hidden_size=128, output_size=1, dropout_rate=0.5)
            criterion = torch.nn.BCEWithLogitsLoss()  
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            early_stopper = EarlyStopper(patience=3, min_delta=0.001)

            for epoch in range(epochs):
                train_acc, train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, max_grad_norm=1.0)
                val_acc, val_loss, val_preds, val_probas, val_targets = _validate_one_epoch(model, val_loader, criterion)
                if epoch % 4 == 0:
                    logging.info(f'Epoch: [{epoch + 1}/{epochs}], Train:: Loss: {train_loss:.4f}, Acc:{train_acc:.4f}, and  Val:: Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ')
                if early_stopper.early_stop(val_loss):             
                    break
            fold_results[fold] = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}
            logging.info(f'Fold {fold+1}/{k_folds} - {fold_results[fold]}')

        avg_val_acc = sum(fold['val_acc'] for fold in fold_results.values()) / k_folds
        avg_val_loss = sum(fold['val_loss'] for fold in fold_results.values()) / k_folds
        logging.info(f'Avg Val Acc: {avg_val_acc} and Val Loss: {avg_val_loss}')
        val_outputs['mlp'][target_col] = (torch.cat(val_targets), torch.cat(val_preds), torch.cat(val_probas))

        #Test
        X_test, y_test, _ = my_train.prepare_dataset(test_dataset.X, test_dataset.contextual_emb, test_dataset.Y[[target_col]], selected_features[target_col])
        y_pred, y_prob = predict(model, X_test)
        test_outputs['mlp'][target_col] = (y_test, y_pred, y_prob)
        logging.info(f'Test Acc: {accuracy_score(y_test, y_pred):.2f}')

    my_train.display_metrics(val_outputs, initial='val')
    my_train.display_metrics(test_outputs, initial='test')
    logging.info(f'{selected_features}')
   
def train(emb, models, demo, kFold, hyperparameters):
    logging.info(f'Training started: emb:{emb} models:{models} demo:{demo} kFold:{kFold}, hyperparameters: {hyperparameters}')
    my_train = My_training(model_list=models, emb_model=emb, demo=demo)
    train_set = Dataset('./processed_data/2-splits/pandora_train_val.csv', my_train.emb_model, my_train.traits, my_train.demo)   
    test_set = Dataset('./processed_data/2-splits/pandora_test.csv', my_train.emb_model, my_train.traits, my_train.demo)    

    logging.info(50*"*")
    selected_features = {}
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        # Scale and Select features
        X, y, selected_features[target_col] = my_train.prepare_dataset(train_set.X, train_set.contextual_emb, train_set.Y[[target_col]], [])

        #train and validate model
        my_train.init_models(X_shape=X.shape[1], kFold=kFold, hyperparameters=hyperparameters)
        my_train.fit_and_validate(X, y, target_col, save_ckpt=False)

        #test model
        X_test, y_test, _ = my_train.prepare_dataset(test_set.X, test_set.contextual_emb, test_set.Y[[target_col]], selected_features[target_col])
        my_train.evaluate_models(X_test, y_test, target_col, calibrate=False)

        #calibrate model 
        # my_train.calibrate_models(X, y, X_test, y_test, target_col)

        logging.info(50*"-")
    my_train.display_metrics(my_train.val_outputs, initial='val')
    my_train.display_metrics(my_train.test_outputs, initial='test')
    # my_train.display_metrics(my_train.cal_outputs, initial='cal')
    # pd.DataFrame(selected_features).to_csv(f"{ckpt}/selected_features.csv")
    logging.info(f'selected_features :{selected_features}')

def final_eval(emb, models, demo, kFold, hyperparameters):
    logging.info(f'Training started: emb:{emb} models:{models} demo:{demo} kFold:{kFold}, hyperparameters: {hyperparameters}')
    my_train = My_training(model_list=models, emb_model=emb, demo=demo)
    train_set = Dataset('./processed_data/2-splits/pandora_train_val.csv', my_train.emb_model, my_train.traits,  my_train.demo)   
    test_set = Dataset('./processed_data/2-splits/pandora_test.csv', my_train.emb_model, my_train.traits, my_train.demo)    
    # my_train.test_df = test_set.ORIGINAL
    logging.info(50*"*")
    selected_features = {}
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        # Scale and Select features
        X, y, selected_features[target_col] = my_train.prepare_dataset(train_set.X, train_set.contextual_emb, train_set.Y[[target_col]], [])
        #train 
        my_train.init_models(X_shape=X.shape[1], kFold=kFold, hyperparameters=hyperparameters)
        my_train.fit(X, y, target_col, save_ckpt=False)
        #test 
        X_test, y_test, _ = my_train.prepare_dataset(test_set.X, test_set.contextual_emb, test_set.Y[[target_col]], selected_features[target_col])
        my_train.evaluate_models(X_test, y_test, target_col, calibrate=False)
    my_train.display_metrics(my_train.test_outputs, initial='final-test')
    # my_train.test_df.to_csv(f'{ckpt}/prediction_test.csv')
    logging.info(f'{selected_features}')
    
if __name__ == "__main__":
    try:
        emb = sys.argv[1]
        models = sys.argv[2]
        kFold = False #sys.argv[3]
        demo = 100
        eval = False #sys.argv[3]
        print(emb, models, kFold, demo)
        emb_models = {'1':'roberta-base', '2':'bert-base-uncased', '3':'vinai/bertweet-base', '4':'xlnet-base-cased'}
        emb = emb_models[emb] if emb in emb_models.keys() else None
        models = ['lr', 'rf', 'xgb', 'mlp', 'bilstm'] if models == 'all' else ['mlp']
        print(emb, models, kFold, demo)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder = f"{emb.split('-')[0]}-{timestamp}" if emb else f"{timestamp}"
        if demo: folder = f"{folder}_demo"
        if eval: folder = f"{folder}_final_eval"
        ckpt = f"checkpoint/{folder}"
        if not os.path.exists(ckpt):
            os.makedirs(f'{ckpt}/calibration/')
        logging.basicConfig(filename=f'{ckpt}/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       
        hyperparameters = {
            'hidden_dim' : 128,
            'dropout_rate' : 0.3,
            'batch_size': 16,
            'epochs': 16,
            'learning_rate': 0.001,
        }
        if eval: final_eval(emb, models, demo, kFold, hyperparameters=hyperparameters) 
        else:
            if kFold: kfold_train(emb, models, demo)
            else: train(emb, models, demo, kFold=kFold, hyperparameters=hyperparameters)

    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
