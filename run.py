
from PersonaClassifier import My_training, parse_arguments, get_filename, set_ckpt
import pandas as pd
import re,os, glob, traceback, nltk, logging, sys
from datetime import datetime
global logging
from utils.DataProcessor import FeatureSelection, PreProcessor
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

class Dataset:
    def __init__(self, filepath, emb_model, targets, demo, task='classification'):
        logging.info(f'Processing {filepath} Dataset.')
        df = pd.read_csv(filepath) 
        if isinstance(demo, int): df = df.sample(demo, random_state=42)
        logging.info(f'{df.shape}')

        df = df[['body', 'agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism', 'type', 'openai_embedding']] # more embedding
        df = df.rename(columns= {'body': 'STATUS'})
        df = PreProcessor.generate_target_labels(df, ['STATUS', 'openai_embedding']+targets)
        df['openai_embedding'] = df['openai_embedding'].apply(eval).apply(np.array)

        df, df_test = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
        self.X_train = df.drop(['STATUS', 'openai_embedding'] + targets, axis=1) # doesn't support statistical features 
        self.Z_train = pd.DataFrame(df["openai_embedding"].to_list(), index=df.index)
        self.Y_train= df[targets]

        self.X_test = df_test.drop(['STATUS', 'openai_embedding'] + targets, axis=1) # doesn't support statistical features 
        self.Z_test = pd.DataFrame(df_test["openai_embedding"].to_list(), index=df_test.index)
        self.Y_test= df_test[targets]
        self.ORIGINAL = df_test[['STATUS', 'openai_embedding'] + targets]
        logging.info(f'X Shape: {self.X_train.shape}, Y Shape: {self.Y_train.shape},  (Z) Shape: {self.Z_train.shape if emb_model else []}')

def split_file_before_run(emb, models, demo, kFold, hyperparameters, filepath, mode):
    my_train = My_training(models=models, emb_model=emb)
    dataset = Dataset(filepath, my_train.emb_model, my_train.traits, demo)  
    logging.info(50*"*")
    epochs = {'cOPN': 17, 'cCON': 13,'cEXT': 16, 'cAGR':17, 'cNEU':15} 
    # my_train.traits = ['cOPN']
    my_train.test_df = dataset.ORIGINAL
    for target_col in my_train.traits:
        logging.info(f'{10*"-"} {target_col} {10*"-"}')
        _, Z_train, y_train = dataset.X_train, dataset.Z_train, dataset.Y_train[[target_col]]
        _, Z_test, y_test = dataset.X_test, dataset.Z_test, dataset.Y_test[[target_col]]
        X_train, X_test = np.array(Z_train), np.array(Z_test)
        y_train, y_test = np.array(y_train).ravel(), np.array(y_test).ravel()
        logging.info(f'X Train: {X_train.shape}, X Test: {X_test.shape} Y Train: {y_train.shape}, Y Test: {y_test.shape}')
        my_train.init_models(X_shape=X_train.shape[1], hyperparameters=hyperparameters, kFold=kFold)
        my_train.fit(X_train, y_train, target_col, save_ckpt=False)
        my_train.evaluate_models(X_test, y_test, target_col, calibrate=False)
    my_train.display_metrics(my_train.test_outputs, initial=f'final_test_kf' if kFold else 'final_test')
    my_train.test_df.to_csv(f'{ckpt}/prediction_test.csv')
    # logging.info(f'{selected_features}')

if __name__ == "__main__":
    try:
        emb, model_type, data_type, mode, evaluate, demo, version, kFold, message, models = parse_arguments(sys.argv)
        filepath = get_filename(data_type, version, emb, demo)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder = f"{data_type}_{version}_{model_type}_{emb.split('-')[0]}-{timestamp}_{evaluate}" if emb else  f"{version}_{model_type}-{timestamp}_{evaluate}"
        if kFold: folder = f"{folder}_kf" 
        if isinstance(demo, int) : folder = f"{folder}_demo"
        ckpt = f"checkpoint/{folder}_{mode}"
        if not os.path.exists(ckpt):
            os.makedirs(f'{ckpt}/calibration/')
            os.makedirs(f'{ckpt}/models/')
        logging.basicConfig(filename=f'{ckpt}/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"emb:{emb}, models:{models}, data_type:{data_type}, mode:{mode}, evaluate:{evaluate}, demo:{demo}, version:{version}, message:{message}, kFold={kFold}")
        hyperparameters = {
            'hidden_dim' : 512,
            'batch_size': 8,
            'epochs': 32,
            'learning_rate': 0.0001,
            'dropout_rate': 0.3,
        }
        logging.info(f"hyperparameters:{hyperparameters}")
        set_ckpt(ckpt)
        split_file_before_run(emb, models, demo, False, hyperparameters, filepath, mode)

    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  


