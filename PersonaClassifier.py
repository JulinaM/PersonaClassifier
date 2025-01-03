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
from utils.Visualization import generate_cm, generate_auroc, display_auroc, display_calibration
from utils.Models import MLP, MLPWrapper, BiLSTMClassifier, IdentityEstimator
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.calibration import CalibratedClassifierCV
global timestamp
global ckpt 
global logging

hyperparameters = {
    'kFold' : False,
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
        self.ORIGINAL = df[['STATUS'] + targets]
        logging.info(f'X Shape: {self.X.shape}, Y Shape: {self.Y.shape},  Contextual Emb Shape: {self.contextual_emb.shape if emb_model else []}')

class My_training:
    def __init__(self, model_list=None, emb_model=None, demo=True,traits=None, ):
        self.models = model_list if model_list else ['svm', 'lr', 'rf', 'xgb', 'bilstm', 'mlp']
        self.emb_model = emb_model
        self.traits = traits if traits else ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU'] 
        self.demo = demo
        self.all_outputs ={}
        self.test_outputs ={}
        self.cal_outputs ={}
        self.test_df = pd.DataFrame()
        for model in model_list:
            self.all_outputs[model] = {}
            self.test_outputs[model] = {}
            self.cal_outputs[model] = {}

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

    def init_models(self, X_shape, kFold):
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
                self.bilstm_Wrapper = MLPWrapper(model=BiLSTMClassifier(input_dim=X_shape, hidden_dim=128, output_dim=1, num_layers=2, bidirectional=True, do_attention=True, dropout_rate=0.5), kFold=kFold )
            elif model == 'mlp':  
                self.mlpWrapper = MLPWrapper(model=MLP(input_size=X_shape, hidden_size=128, output_size=1, dropout_rate=0.5), kFold=kFold)
        logging.info(f'Model Initiated.')
    
    def fit_models(self, X, y, target_col, save_ckpt=False):
        logging.info(f'Fitting and Validating Models...')
        for model in self.models:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
            if model =='svm':
                self.svm_model.fit(X_train, y_train)
                pred, probs = self.svm_model.predict(X_val), self.svm_model.decision_function(X_val)[:, 1]
            elif model == 'lr':
                self.lr_model.fit(X_train, y_train)
                pred, probs = self.lr_model.predict(X_val), self.lr_model.predict_proba(X_val)[:, 1]
            elif model == 'rf':
                self.rf_model.fit(X_train, y_train)
                pred, probs = self.rf_model.predict(X_val), self.rf_model.predict_proba(X_val)[:, 1]
            elif model == 'xgb': 
                self.xgb_model.fit(X_train, y_train)
                pred, probs = self.xgb_model.predict(X_val), self.xgb_model.predict_proba(X_val)[:, 1]
                if save_ckpt: self.xgb_model.save_model(f"{ckpt}/{self.xgb_model.__class__.__name__}_{target_col}.json")
            elif model == 'bilstm':   
                val_acc, pred, probs, y_val = self.bilstm_Wrapper.fit(X, y)
                if save_ckpt: torch.save(self.bilstm_model.state_dict(), f"{ckpt}/{self.bilstm_model.__class__.__name__}_{target_col}.pth")
            elif model == 'mlp':  
                val_acc, pred, probs, y_val = self.mlpWrapper.fit(X, y)
                if save_ckpt: torch.save(self.mlpWrapper.model.state_dict(), f"{ckpt}/{self.mlpWrapper.model.__class__.__name__}_{target_col}.pth")
            self.all_outputs[model][target_col] = (y_val, pred, probs)
            logging.info(f'{model} Val Acc: {accuracy_score(y_val, pred):.2f}')
        logging.info(f'Model Fitted.')

    def evaluate_models(self, X, y, target_col):
        logging.info(f'Evaluating on Test Dataset.')
        for model in self.models:
            if model =='svm':
                pred, probs = self.svm_model.predict(X), self.svm_model.decision_function(X)[:, 1]
            elif model == 'lr':
                pred, probs = self.lr_model.predict(X), self.lr_model.predict_proba(X)[:, 1]
            elif model == 'rf':
                pred, probs = self.rf_model.predict(X), self.rf_model.predict_proba(X)[:, 1]
            elif model == 'xgb': 
                pred, probs = self.xgb_model.predict(X), self.xgb_model.predict_proba(X)[:, 1]
            elif model == 'bilstm':   
                pred, probs = self.bilstm_Wrapper.predict(X), self.bilstm_Wrapper.predict_proba(X)
            elif model == 'mlp':  
                pred, probs = self.mlpWrapper.predict(X), self.mlpWrapper.predict_proba(X)
            # display_calibration(y, probs, target_col, f'{ckpt}/calibration/{model}_{target_col}.png')
            generate_cal_result(y, pred, probs, target_col, f'{ckpt}/calibration/uncal_{model}_{target_col}.png')
            self.test_outputs[model][target_col] = (y, pred, probs)
            self.test_df[f'{model}_{target_col}'] = pred
            logging.info(f'{model} Test Acc: {accuracy_score(y, pred):.2f}')
        logging.info(f'Evaluation completed.')


    def calibrate_models(self, X, y, X_test, y_test, target_col):
        for model in self.models:
            y_prob_val = self.mlpWrapper.predict_proba(X)
            y_prob_test = self.mlpWrapper.predict_proba(X_test)
            calibrated = CalibratedClassifierCV(base_estimator = IdentityEstimator(), method = 'sigmoid', cv = 5)
            calibrated.fit(y_prob_val, y)
            y_cal_pred, y_cal_prob = calibrated.predict(y_prob_test), calibrated.predict_proba(y_prob_test)[:, 1]
            generate_cal_result(y_test, y_cal_pred, y_cal_prob, target_col, f'{ckpt}/calibration/cal_{model}_{target_col}.png' )
            self.cal_outputs[model][target_col] = (y_test, y_cal_pred, y_cal_prob)
        

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

def generate_cal_result(y_true, y_pred, y_prob, target_col, filename):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
    from sklearn.metrics import roc_curve, auc, roc_auc_score, brier_score_loss
    clf_score = brier_score_loss(y_true, y_prob, pos_label=1)
    logging.info(f"{filename}")
    logging.info("\tBrier: %1.3f" % (clf_score))
    logging.info("\tAccuracy: %1.3f" % accuracy_score(y_true, y_pred))
    logging.info("\tPrecision: %1.3f" % precision_score(y_true, y_pred))
    logging.info("\tRecall: %1.3f" % recall_score(y_true, y_pred))
    logging.info("\tF1: %1.3f" % f1_score(y_true, y_pred))
    display_calibration(y_true, y_prob, target_col, filename)

def kfold_train(emb, models, demo):
    from skorch import NeuralNetClassifier
    logging.info(f'K-Fold Training started: {emb} {models} {demo}')
    my_train = My_training(model_list=models, emb_model=emb, demo=demo)
    dataset = Dataset('./processed_data/2-splits/pandora_train_val.csv', my_train.emb_model, my_train.traits, my_train.demo)    
    test_dataset = Dataset('./processed_data/2-splits/pandora_test.csv', my_train.emb_model, my_train.traits, my_train.demo)    
    my_train.test_df = test_dataset.ORIGINAL
    logging.info(50*"*")
    train_outtputs, test_outputs, cal_test_outputss, selected_features = {'mlp':{}}, {'mlp':{}}, {'mlp':{}}, {}
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        #feature selection
        features = my_train.select_features(dataset.X, dataset.Y[[target_col]])
        selected_features[target_col] = features
        logging.info(f'Selected Features for {target_col} : {len(features)}')

        #Train
        X, y = my_train.prepare_dataset(dataset.X[features], dataset.contextual_emb, dataset.Y[[target_col]])
        mlp = MLP(input_size=X.shape[1], hidden_size=128, output_size=1, dropout_rate=0.5)
        mlpWrapper = MLPWrapper(model=mlp, kFold=True)
        _, y_pred, y_prob, y_val = mlpWrapper.fit(X, y)
        y_prob_val = mlpWrapper.predict_proba(X)
        train_outtputs['mlp'][target_col] = (y_val, y_pred, y_prob)
        logging.info(f"Val Accuracy: {target_col}: {accuracy_score(y_val, y_pred)}")

        # #Evaluate
        X_test, y_test = my_train.prepare_dataset(test_dataset.X[features], test_dataset.contextual_emb, test_dataset.Y[[target_col]])
        y_pred, y_prob = mlpWrapper.predict(X_test), mlpWrapper.predict_proba(X_test)
        test_outputs['mlp'][target_col] = (y_test, y_pred, y_prob)
        my_train.test_df[f'mlp_uncal_prob_{target_col}'] = y_pred
        generate_cal_result(y_test, y_pred, y_prob, target_col, f'{ckpt}/calibration/uncal_{target_col}.png' )
        logging.info(f"Test Accuracy: {target_col}: {accuracy_score(y_test, y_pred)}")

        # #Calibrate
        from sklearn.calibration import CalibratedClassifierCV
        calibrated = CalibratedClassifierCV(base_estimator = IdentityEstimator(), method = 'sigmoid', cv = 5)
        calibrated.fit(y_prob_val, y)
        y_cal_pred, y_cal_prob = calibrated.predict(y_prob), calibrated.predict_proba(y_prob)[:, 1]
        generate_cal_result(y_test, y_cal_pred, y_cal_prob, target_col, f'{ckpt}/calibration/cal_{target_col}.png' )
        cal_test_outputss['mlp'][target_col] = (y_test, y_cal_pred, y_cal_prob)
        my_train.test_df[f'mlp_cal_prob_{target_col}'] = y_cal_prob
        
    my_train.display_metrics(train_outtputs, initial='val')
    my_train.display_metrics(test_outputs, initial='test')
    my_train.display_metrics(cal_test_outputss, initial='cal')
    # pd.DataFrame(selected_features).to_csv(f"{ckpt}/selected_features.csv")
    my_train.test_df.to_csv(f'{ckpt}/prediction_test.csv')
    logging.info(f'selected_features :{selected_features}')
   
def train(emb, models, demo, kFold):
    logging.info(f'Training started: emb:{emb} models:{models} demo:{demo} kFold:{kFold}')
    my_train = My_training(model_list=models, emb_model=emb, demo=demo)
    train_set = Dataset('./processed_data/rd_fb/pandora_train_val.csv', my_train.emb_model, my_train.traits, my_train.demo)   
    test_set = Dataset('./processed_data/rd_fb/pandora_test.csv', my_train.emb_model, my_train.traits, my_train.demo)    
    my_train.test_df  = test_set.ORIGINAL

    logging.info(50*"*")
    selected_features ={}
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        #feature_selection
        features = my_train.select_features(train_set.X, train_set.Y[[target_col]] )
        selected_features[target_col] = features
        logging.info(f'Selected Features for {target_col} : {len(features)}')

        #train and validate model
        X, y = my_train.prepare_dataset(train_set.X[features], train_set.contextual_emb, train_set.Y[[target_col]])
        my_train.init_models(X_shape=X.shape[1], kFold=kFold)
        my_train.fit_models(X, y, target_col, save_ckpt=False)

        #test model
        X_test, y_test = my_train.prepare_dataset(test_set.X[features], test_set.contextual_emb, test_set.Y[[target_col]])
        my_train.evaluate_models(X_test, y_test, target_col)

        my_train.calibrate_models(X, y, X_test, y_test, target_col)
        
        logging.info(50*"-")
    my_train.display_metrics(my_train.all_outputs,  initial='val')
    my_train.display_metrics(my_train.test_outputs, initial='test')
    my_train.display_metrics(my_train.cal_outputs, initial='cal')
    # my_train.test_df.to_csv(f'{ckpt}/prediction_test.csv')
    # pd.DataFrame(selected_features).to_csv(f"{ckpt}/selected_features.csv")
    logging.info(f'selected_features :{selected_features}')

    
if __name__ == "__main__":
    try:
        emb = sys.argv[1]
        models = sys.argv[2]
        kFold = False #sys.argv[3]
        demo = None
        print(emb, models, kFold, demo)
        emb_models = {'1':'roberta-base', '2':'bert-base-uncased', '3':'vinai/bertweet-base', '4':'xlnet-base-cased'}
        emb = emb_models[emb] if emb in emb_models.keys() else None
        models = ['lr', 'rf', 'xgb', 'mlp', 'bilstm'] if models == 'all' else [ 'mlp']
        print(emb, models, kFold, demo)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ckpt = f"checkpoint/{emb.split('-')[0]}-{timestamp}" if emb else f"checkpoint/{timestamp}"
        if not os.path.exists(ckpt):
            os.makedirs(f'{ckpt}/calibration/')
        logging.basicConfig(filename=f'{ckpt}/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       
        if kFold: kfold_train(emb, models, demo)
        else: train(emb, models, demo, kFold=kFold)

    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
