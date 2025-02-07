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
from utils.Models import MLP, MyEstimator, BiLSTMClassifier, IdentityEstimator
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from utils.Training import _train_one_epoch, _validate_one_epoch, predict, EarlyStopper
from torch.utils.data import DataLoader, TensorDataset, Subset
import shap
shap.initjs()

global ckpt 
global logging
# big_5_traits = ['agreeableness', 'openness', 'conscientiousness', 'extraversion','neuroticism']
class ModelCreator:
    def __init__(self, X_shape, kFold, hyperparameters):
        hyperparameters['input_dim'] = X_shape
        logging.info(f'Model Creator initiated. hyperparameters: {hyperparameters}.')

        self.estimators = {
            "svm" : SVC(kernel='linear'),
            "lr" : LogisticRegression(solver='lbfgs', max_iter=1000),
            "rf" : RandomForestClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
            'mlp': MyEstimator(model_name="mlp", input_dim=hyperparameters['input_dim'], hidden_dim=hyperparameters['hidden_dim'], dropout_rate=hyperparameters['dropout_rate'], batch_size=hyperparameters['batch_size'], epochs = hyperparameters['epochs'], lr=hyperparameters['learning_rate'], kFold=False),
            'bilstm': MyEstimator(model_name="bilstm", input_dim=hyperparameters['input_dim'], hidden_dim=hyperparameters['hidden_dim'], dropout_rate=hyperparameters['dropout_rate'], batch_size=hyperparameters['batch_size'], epochs = hyperparameters['epochs'], lr=hyperparameters['learning_rate'], kFold=False)
        }

class Dataset:
    def __init__(self, filepath, emb_model, targets, demo, version=None):
        logging.info(f'Processing {filepath} Dataset.')
        df = pd.read_csv(filepath) 
        if isinstance(demo, int): df = df.sample(demo, random_state=42)
        logging.info(f'{df.shape}')
        if filepath.split('_')[-1] =='main.csv': 
            df = df[['author', 'body', 'agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']] # more embedding
            df = df.rename(columns= { 'body': 'STATUS', 'agreeableness':'cAGR', 'openness':'cOPN', 'conscientiousness':'cCON', 'extraversion':'cEXT', 'neuroticism':'cNEU'})
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.loc[:, ~df.columns.str.contains('^#AUTHID')]
        PreProcessor.clean_up_text(df)
        logging.info(df.head(1))
        self.contextual_emb = PreProcessor.process_embeddings(df, emb_model) if emb_model else []
        self.X = df.drop(['STATUS'] + targets, axis=1) #remove #AUTHID for V1
        self.Y = df[targets]
        self.ORIGINAL = df[['STATUS'] + targets]
        logging.info(f'X Shape: {self.X.shape}, Y Shape: {self.Y.shape}, Contextual Emb (Z) Shape: {self.contextual_emb.shape if emb_model else []}')

class My_training:
    def __init__(self, models=None, emb_model=None, traits=None):
        logging.info(f'Training initiated: emb:{emb_model} models:{models}')
        self.models = models if models else ['svm', 'lr', 'rf', 'xgb', 'bilstm', 'mlp']
        self.emb_model = emb_model
        self.traits = traits if traits else ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU'] 
        self.test_outputs ={}
        self.val_outputs = {}
        self.cal_outputs ={}
        self.estimators = {}
        for model in models:
            self.test_outputs[model] = {}
            self.val_outputs[model]= {}
            self.cal_outputs[model] = {}
        self.test_df = pd.DataFrame()

    def scale_features(self, X):
        scaler = StandardScaler() 
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X
        
    def feature_extraction(self, X, y, corr_thres=0.25, k=10):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_thres)]
        X_reduced = pd.DataFrame(X).drop(to_drop, axis=1)
        # fs = X_reduced.columns
        logging.info(f'Correlation Threshold: {corr_thres} dropped: {len(to_drop)}')
        logging.info(f'Reduced to: {X_reduced.shape} {X_reduced.columns}')
        fs = FeatureSelection.filter_selection(X_reduced, y, k)
        logging.info(f'Feature Selected: {fs}')
        return fs

    def combine_dataset(self, stat_df, emb_df, y_df):
        logging.info(f'Combining {stat_df.shape}, {emb_df.shape}, {y_df.shape}')
        X = np.concatenate([stat_df, emb_df], axis=1)
        X = np.array(X)
        y = np.array(y_df).ravel()
        logging.info(f'Final shape X and y: {X.shape} and {y.shape}')
        return X, y

    def init_models(self, X_shape, kFold, hyperparameters):
        model_creator = ModelCreator(X_shape, kFold, hyperparameters)
        for model in self.models:
            self.estimators[model] = model_creator.estimators[model]
        logging.info(f'Model Initiated.')
    
    def fit_and_validate(self, X_train, y_train, X_val, y_val, target_col, save_ckpt=False):
        logging.info(f'Fitting and Validating Models...')
        for model in self.models:
            estimator = self.estimators[model]
            if model in ['mlp', 'bilstm']:
                self.estimators[model].set_val_data(X_val, y_val)
            estimator.fit(X_train, y_train)
            y_pred, y_prob = estimator.predict(X_val), estimator.predict_proba(X_val)[:,1]
            self.val_outputs[model][target_col] = (y_val, y_pred, y_prob)
            logging.info(f'{model} Val Acc: {accuracy_score(y_val, y_pred):.2f}')
        logging.info(f'Model Fitted and Validated.')

    def fit(self, X, y, target_col, save_ckpt=False):
        logging.info(f'Fitting Models...')
        for model in self.models:
            estimator = self.estimators[model]
            estimator.fit(X, y)
            if save_ckpt and  model not in ['lr', 'rf']: estimator.save_model(f"{ckpt}/models/{estimator.__class__.__name__}_{target_col}.json")
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
        dfs = []
        for model in all_outputs:
            logging.info(15*'='+f" {model} "+ 15*'=')
            a_output = all_outputs[model]
            (cm, auroc, perf) = (f'cm_{initial}', f'auroc_{initial}', f'performance_{initial}') if initial else ('cm', 'auroc', 'performance')
            data = generate_cm(a_output, f'{ckpt}/{model}_{cm}.png')
            data = [dict(d, **{'model': model}) for d in data]
            df = pd.DataFrame(data)
            dfs.append(df)
            generate_auroc(a_output, model, f'{ckpt}/{model}_{auroc}.png')
        performance_df = pd.concat(dfs, axis=0)
        logging.info(f"Performance df shape: {performance_df.shape}")
        if savefig: performance_df.to_csv(f"{ckpt}/{perf}.csv")
            # # for col in self.traits:
            #     s = performance_df[performance_df['Classifier'] ==col]
            #     best_model_row = s.loc[s['Accuracy'].idxmax()]
            #     logging.info(f'For {best_model_row["Classifier"]}, {best_model_row["Model"]},  {best_model_row["Accuracy"]}')

def kfold_train(emb, models, demo, filepath):
    logging.info(f'K-Fold Training started: {emb} {models} {demo}')
    my_train = My_training(models=models, emb_model=emb)
    dataset = Dataset('./processed_data/2-splits/pandora_train_val.csv', my_train.emb_model, my_train.traits, demo)    
    test_dataset = Dataset('./processed_data/2-splits/pandora_test.csv', my_train.emb_model, my_train.traits, demo)    
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

def train(emb, models, demo, kFold, hyperparameters, filepath, mode):
    my_train = My_training(models=models, emb_model=emb)
    dataset = Dataset(filepath, my_train.emb_model, my_train.traits,  demo)   

    logging.info(50*"*")
    selected_features = {}
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        X, Z, y = dataset.X, dataset.contextual_emb, dataset.Y[[target_col]]
        if mode == "S0":
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=42)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,  stratify=y_test, test_size=0.5, shuffle=True, random_state=42)
            X_train, X_val, X_test  = my_train.scale_features(X_train), my_train.scale_features(X_val), my_train.scale_features(X_test)
            X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
            y_train, y_val, y_test = np.array(y_train).ravel(), np.array(y_val).ravel(), np.array(y_test).ravel()
        elif mode == "S1":
            Z_train, Z_test, y_train, y_test, = train_test_split(Z, y, stratify=y, test_size=0.2, shuffle=True, random_state=42)
            Z_val, Z_test, y_val, y_test, = train_test_split(Z_test, y_test, stratify=y_test, test_size=0.5, shuffle=True, random_state=42)
            X_train, X_val, X_test = np.array(Z_train), np.array(Z_val), np.array(Z_test)
            y_train, y_val, y_test = np.array(y_train).ravel(), np.array(y_val).ravel(), np.array(y_test).ravel()
        elif mode == "S2":
            X_train, X_test, Z_train, Z_test, y_train, y_test, = train_test_split(X, Z, y, stratify=y, test_size=0.2, shuffle=True, random_state=42)
            X_test, X_val, Z_test, Z_val, y_test, y_val = train_test_split(X_test, Z_test, y_test, stratify=y_test, test_size=0.5, shuffle=True, random_state=42)
            X_train, X_val, X_test  = my_train.scale_features(X_train), my_train.scale_features(X_val), my_train.scale_features(X_test)
            X_train, y_train = my_train.combine_dataset(X_train, Z_train, y_train)
            X_val, y_val = my_train.combine_dataset(X_val, Z_val, y_val)
            X_test, y_test = my_train.combine_dataset(X_test, Z_test, y_test)
        elif mode == "S3" :
            X_train, X_test, Z_train, Z_test, y_train, y_test, = train_test_split(X, Z, y, stratify=y, test_size=0.2, shuffle=True, random_state=42)
            X_test, X_val, Z_test, Z_val, y_test, y_val = train_test_split(X_test, Z_test, y_test, stratify=y_test, test_size=0.5, shuffle=True, random_state=42)
            X_train, X_val, X_test  = my_train.scale_features(X_train), my_train.scale_features(X_val), my_train.scale_features(X_test)
            sf = my_train.feature_extraction(X_train, y_train, corr_thres=0.25) # feature reduction/selection
            selected_features[target_col] = sf
            X_train, y_train = my_train.combine_dataset(X_train[sf], Z_train, y_train)
            X_val, y_val = my_train.combine_dataset(X_val[sf], Z_val, y_val)
            X_test, y_test = my_train.combine_dataset(X_test[sf], Z_test, y_test)

        #train and validate model
        my_train.init_models(X_shape=X_train.shape[1], kFold=kFold, hyperparameters=hyperparameters)
        my_train.fit_and_validate(X_train, y_train, X_val, y_val, target_col, save_ckpt=False)
        #test model
        my_train.evaluate_models(X_test, y_test, target_col, calibrate=False)
        #calibrate model s
        # my_train.calibrate_models(X_val, y_val, X_test, y_test, target_col)
        logging.info(50*"-")
    my_train.display_metrics(my_train.val_outputs, initial='val')
    my_train.display_metrics(my_train.test_outputs, initial='test')
    # my_train.display_metrics(my_train.cal_outputs, initial='cal')
    # pd.DataFrame(selected_features).to_csv(f"{ckpt}/selected_features.csv")
    logging.info(f'selected_features :{selected_features}')

def regression_train(emb, models, demo, kFold, hyperparameters, filepath, mode, version):
    my_train = My_training(models=models, emb_model=emb)
    dataset = Dataset(filepath, my_train.emb_model, my_train.traits,  demo, version)   
    logging.info(50*"*")
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        X, Z, y = dataset.X, dataset.contextual_emb, dataset.Y[[target_col]]
        if mode == "S1":
            Z_train, Z_test, y_train, y_test, = train_test_split(Z, y, test_size=0.2, shuffle=True, random_state=42)
            Z_val, Z_test, y_val, y_test, = train_test_split(Z_test, y_test, test_size=0.5, shuffle=True, random_state=42)
            X_train, X_val, X_test = np.array(Z_train), np.array(Z_val), np.array(Z_test)
            y_train, y_val, y_test = np.array(y_train).ravel(), np.array(y_val).ravel(), np.array(y_test).ravel()

        logging.info(f"{X_train.shape}, {X_val.shape}, {X_test.shape}")
        from utils.RegressionModels import RegressionModel, train_val, evaluate_model, plot_learning_curve, BiLSTMClassifier
        import torch.optim as optim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = BiLSTMClassifier(input_dim=X_train.shape[1], hidden_dim=hyperparameters["hidden_dim"], dropout=hyperparameters["dropout_rate"]).to(device)
        model = RegressionModel(input_dim=X_train.shape[1], hidden_dim=hyperparameters["hidden_dim"], dropout=hyperparameters["dropout_rate"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
        train_losses, val_losses = train_val(model, X_train, y_train, X_val, y_val,
            device=device, batch_size=hyperparameters["batch_size"], epochs=hyperparameters["epochs"], optimizer=optimizer, max_grad_norm=1.0
        )
        plot_learning_curve(train_losses, val_losses, f"{ckpt}/learning_curve_{target_col}.png")
        logging.info(f'Final Validation RMSE: {val_losses[-1]:.4f}')
        _, _, mse, mae, r2, _, _ = evaluate_model(model, X_test, y_test, device)
        logging.info(f"Test MSE: {mse:.4f}")

def final_eval(emb, models, demo, kFold, hyperparameters, filepath, mode):
    my_train = My_training(models=models, emb_model=emb)
    dataset = Dataset(filepath, my_train.emb_model, my_train.traits,  demo)   
    if mode == 'T': test_dataset = Dataset(f'./processed_data/LIWC_mypersonality_v2.csv', my_train.emb_model, my_train.traits,  demo)   
    logging.info(50*"*")
    selected_features = {}
    epochs = {'cOPN': 17, 'cCON': 13,'cEXT': 16, 'cAGR':17, 'cNEU':15} 
    # my_train.test_df = test_dataset.ORIGINAL
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        X, Z, y = dataset.X, dataset.contextual_emb, dataset.Y[[target_col]]
        if mode == "S0":
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, shuffle=True, random_state=42)
            X_train, X_test = my_train.scale_features(X_train), my_train.scale_features(X_test)
            X_train, X_test = np.array(X_train), np.array(X_test)
            y_train, y_test = np.array(y_train).ravel(), np.array(y_test).ravel()
        elif mode == "S1":
            Z_train, Z_test, y_train, y_test, = train_test_split(Z, y, stratify=y, test_size=0.1, shuffle=True, random_state=42)
            X_train, X_test = np.array(Z_train), np.array(Z_test)
            y_train, y_test = np.array(y_train).ravel(), np.array(y_test).ravel()
        elif mode == "S2":
            X_train, X_test, Z_train, Z_test, y_train, y_test, = train_test_split(X, Z, y, stratify=y, test_size=0.1, shuffle=True, random_state=42)
            X_train, X_test = my_train.scale_features(X_train), my_train.scale_features(X_test)
            X_train, y_train = my_train.combine_dataset(X_train, Z_train, y_train)
            X_test, y_test = my_train.combine_dataset(X_test, Z_test, y_test)
        elif mode == "S3" :
            X_train, X_test, Z_train, Z_test, y_train, y_test, = train_test_split(X, Z, y, stratify=y, test_size=0.1, shuffle=True, random_state=42)
            X_train, X_test = my_train.scale_features(X_train), my_train.scale_features(X_test)
            sf = selected_features[target_col]
            X_train, y_train = my_train.combine_dataset(X_train[sf], Z_train, y_train)
            X_test, y_test = my_train.combine_dataset(X_test[sf], Z_test, y_test)
        elif mode == "T":
            X_train, Z_train, y_train = dataset.X, dataset.contextual_emb, dataset.Y[[target_col]]
            X_test, Z_test, y_test = test_dataset.X, test_dataset.contextual_emb, test_dataset.Y[[target_col]]
            X_train, X_test = np.array(Z_train), np.array(Z_test)
            y_train, y_test = np.array(y_train).ravel(), np.array(y_test).ravel()
        logging.info(f'X Train: {X_train.shape}, X Test: {X_test.shape} Y Train: {y_train.shape}, Y Test: {y_test.shape}')

        hyperparameters['epochs'] = epochs[target_col]
        my_train.init_models(X_shape=X_train.shape[1], kFold=kFold, hyperparameters=hyperparameters)
        my_train.fit(X_train, y_train, target_col, save_ckpt=True)
        my_train.evaluate_models(X_test, y_test, target_col, calibrate=True)

        # #shap evaluation
        # explainer = shap.Explainer(my_train.estimators['xgb'], X_train, feature_names=sf)
        # shap.plots.beeswarm(explainer(X_test))
        # # shap_values = explainer.shap_values(X_test[sf]) 
        # # shap.summary_plot(shap_values, X_test[sf], feature_names=sf, max_display=5)
        # plt.savefig(f"{ckpt}/bee_swarm_{target_col}.png", dpi=150, bbox_inches='tight')

    my_train.display_metrics(my_train.test_outputs, initial='final-test')
    # my_train.test_df.to_csv(f'{ckpt}/prediction_test.csv')
    logging.info(f'{selected_features}')
    
def parse_arguments(argv):
    def convert_to_int_if_possible(value):
        try:
            return int(value)
        except ValueError:
            return value
    print(argv)
    emb = argv[1]
    model_type = argv[2]
    data_type = argv[3]
    mode = argv[4]
    eval = argv[5] 
    demo = argv[6]
    version = argv[7]
    kFold = False 
    message = argv[8]
    emb_models = {'1':'roberta-base', '2':'bert-base-uncased', '3':'vinai/bertweet-base', '4':'xlnet-base-cased'}
    emb = emb_models[emb] if emb in emb_models.keys() else None

    models = ['lr', 'rf', 'xgb', 'bilstm', 'mlp']
    if model_type != "all" and model_type not in models:
        print(f"not a valid modeltype {model_type}")
        exit(0)  

    models = models if model_type=='all' else [model_type]
    if demo=="demo": demo =100
    demo = convert_to_int_if_possible(demo)
    print(f"emb:{emb}, models:{models}, data_type:{data_type}, mode:{mode}, eval:{eval}, demo:{demo}, version:{version}, message:{message}, kFold={kFold}")
    return emb, model_type, data_type, mode, eval, demo, version, kFold, message, models
 
if __name__ == "__main__":
    try:
        emb, model_type, data_type, mode, eval, demo, version, kFold, message, models = parse_arguments(sys.argv)
        if data_type =='fb':
            version = 'v2'
            filepath = f'./processed_data/LIWC_mypersonality_{version}.csv' 
        else:
            if version =='v5' or version =='v3': 
                filepath = f'./data/pandora_to_big5_main.csv'  
            else:
                filepath = f'./data/pandora_to_big5_{version}.csv'  

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder = f"{data_type}_{version}_{model_type}_{emb.split('-')[0]}-{timestamp}_{eval}" if emb else  f"{version}_{model_type}-{timestamp}_{eval}"
        if isinstance(demo, int) : folder = f"{folder}_demo"
        ckpt = f"checkpoint/{folder}_{mode}"
        if not os.path.exists(ckpt):
            os.makedirs(f'{ckpt}/calibration/')
            os.makedirs(f'{ckpt}/models/')
        logging.basicConfig(filename=f'{ckpt}/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"emb:{emb}, models:{models}, data_type:{data_type}, mode:{mode}, eval:{eval}, demo:{demo}, version:{version}, message:{message}, kFold={kFold}")
        hyperparameters = {
            'hidden_dim' : 300,
            'batch_size': 8,
            'epochs': 100,
            'learning_rate': 0.00001,
            'dropout_rate': 0.3,
        }
        logging.info(f"hyperparameters:{hyperparameters}")

        if version =='v5': #regresssion
            regression_train(emb, models, demo, kFold=kFold, hyperparameters=hyperparameters, filepath=filepath, mode=mode, version=version)
            sys.exit(1) 

        if eval == 'eval': 
            final_eval(emb, models, demo, kFold, hyperparameters=hyperparameters, filepath=filepath, mode=mode)
        else: # eval == 'train'
            if kFold: kfold_train(emb, models, demo, filepath =filepath)
            else: train(emb, models, demo, kFold=kFold, hyperparameters=hyperparameters, filepath=filepath, mode=mode)

    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
