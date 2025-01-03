import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re,os, glob, traceback, nltk, logging, sys
from datetime import datetime
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
global timestamp
global ckpt 
global logging

class DataProcessor:
    def read_data(main_file, liwc_file, is_mypersonality=True):
        if is_mypersonality:
            main_df = pd.read_csv(main_file, encoding='Windows-1252')
            main_df.drop(columns=['#AUTHID', 'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN', 'DATE', 'NETWORKSIZE', 'BETWEENNESS', 'NBETWEENNESS', 'DENSITY', 'BROKERAGE', 'NBROKERAGE','TRANSITIVITY'], inplace=True)
            main_df[['cOPN', 'cEXT', 'cNEU', 'cAGR', 'cCON']] = main_df[['cOPN', 'cEXT', 'cNEU', 'cAGR', 'cCON']].replace({'y': 1, 'n': 0})
            liwc_df = pd.read_csv(liwc_file)
            liwc_df = liwc_df.drop(['Unnamed: 0', '#AUTHID', 'ColumnID', 'STATUS'], axis=1)
        else:
            main_df = pd.read_csv(main_file)
            main_df.drop(columns=['#AUTHID'], inplace=True)
            cols = ['cOPN', 'cEXT', 'cNEU', 'cAGR', 'cCON']
            for col in cols:
                mean_value = main_df[col].mean()
                main_df[f'{col}'] = main_df[col] > mean_value
            main_df[cols] = main_df[cols].replace({True: 1, False: 0})  
            liwc_df = pd.read_csv(liwc_file)
            liwc_df = liwc_df.drop(['Unnamed: 0', '#AUTHID', 'STATUS', 'cOPN', 'cEXT', 'cNEU', 'cAGR', 'cCON', ], axis=1)

        logging.info(f'File={main_file} shape={main_df.shape}')
        logging.info(f'File={liwc_file} shape={liwc_df.shape}')

        df = pd.concat([main_df, liwc_df], axis=1)
        logging.info(f'Merged main and liwc files, Shape:{df.shape}')
        return df

    def process_NRC_VAD(df):
        nrc_vad = pd.read_csv('data/NRC-VAD-Lexicon/NRC-VAD-Lexicon.csv', sep="\t")  
        nrc_vad_dict = nrc_vad.set_index('Word').to_dict(orient='index')
        def get_vad_scores(text):
            words = text.split()
            valence_scores, arousal_scores, dominance_scores = [], [], []
            for word in words:
                word = word.lower()  # Lowercase to match the lexicon
                if word in nrc_vad_dict:
                    vad_values = nrc_vad_dict[word]
                    valence_scores.append(vad_values['Valence'])
                    arousal_scores.append(vad_values['Arousal'])
                    dominance_scores.append(vad_values['Dominance'])
            if not valence_scores:
                return {'Valence': 0, 'Arousal': 0, 'Dominance': 0}

            valence_avg = sum(valence_scores) / len(valence_scores)
            arousal_avg = sum(arousal_scores) / len(arousal_scores)
            dominance_avg = sum(dominance_scores) / len(dominance_scores)
            return {'Valence': valence_avg, 'Arousal': arousal_avg, 'Dominance': dominance_avg}

        df['VAD_Scores'] = df['STATUS'].apply( lambda x: get_vad_scores(x))
        df[['Valence', 'Arousal', 'Dominance']] = pd.DataFrame(df['VAD_Scores'].tolist(), index=df.index)
        df.drop(columns=['VAD_Scores'], inplace=True)
        logging.info(f'NRC-VAD shape={df.shape}')
        return df

    def process_NRC_emotion(df):
        nrc_lexicon = pd.read_csv('data/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', names=["word", "emotion", "association"],sep="\t", header=None)
        # Filter out words that have no association with emotions (association == 0)
        nrc_lexicon = nrc_lexicon[nrc_lexicon['association'] == 1]
        # nrc_lexicon.drop(columns=['association'], inplace=True)
        nrc_pivot = nrc_lexicon.pivot(index="word", columns="emotion", values="association").fillna(0).astype(int)
        # nrc_pivot.head(2)
        nltk.download('punkt')
        def get_emotion_counts(text, lexicon):
            # print(text)
            words = nltk.word_tokenize(text.lower())
            emotion_count = defaultdict(int)
            for word in words:
                if word in lexicon.index:
                    for emotion in lexicon.columns:
                        emotion_count[emotion] += lexicon.loc[word, emotion]
            return emotion_count
        emotion_counts_list = df['STATUS'].apply(lambda x: get_emotion_counts(x, nrc_pivot))
        emotion_counts_df = pd.DataFrame(emotion_counts_list.tolist())
        emotion_counts_df.fillna(0, inplace=True)
        emotion_counts_df = emotion_counts_df.astype(int)
        df = pd.concat([df, emotion_counts_df], axis=1)
        logging.info(f'NRC-Emotion shape={df.shape}')
        return df

    def process_VADER_sentiment(df):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        def find_sentiment(text):
            # print(text)
            vs = analyzer.polarity_scores(text)
            sc = vs['compound']
            # emo = 'pos' if sc >= 0.05 else 'neu' if -0.05 < sc < 0.05 else 'neg'
            return sc
        df[['sent_score']] = df['STATUS'].apply(lambda x: pd.Series(find_sentiment(x)))
        logging.info(f'VADER shape={df.shape}')
        return df

    def clean_up_text(df):
        logging.info(f'Before cleaning up: {df.shape}')
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        # df['STATUS'] = df['STATUS'].apply(preprocess_text)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed:')]
        df= df.drop_duplicates(subset='STATUS', keep='first')
        logging.info(f"After Dropping duplicate: {df.shape}")
        df['STATUS'] = df['STATUS'].astype(str).fillna('')
        df = df[df['STATUS'].str.strip() != '']
        df = df[df['STATUS'].str.split().str.len() >= 3]
        logging.info(f"After Dropping less than 3 words: {df.shape}")
        logging.info(f"Inserting mean value for null values.")
        num_cols = df.select_dtypes(include=['number']).columns
        #Imputation
        threshold_value = 0.8
        df = df[df.columns[df.isnull().mean() < threshold_value]]         #Dropping columns with missing value rate higher than threshold
        # df = df.loc[df.isnull().mean(axis=1) < threshold_value]        #Dropping rows with missing value rate higher than threshold
        #Numerical Imputation
        # df.fillna(df.mean(), inplace=True)
        # df.fillna(df.median(), implace=True)
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        logging.info(f"After Clean up: {df.shape}")
        return df

    def process_embeddings(df, model_name, batch_size=8):
        from transformers import AutoTokenizer, AutoModel
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f'Generating Embedding from {model_name}')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()  
        embeddings_list = []
        
        for i in range(0, len(df), batch_size):
            batch_texts = df['STATUS'][i:i + batch_size].tolist()
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings_list.append(cls_embeddings.cpu().numpy())
        logging.info(f'Embedding Completed for {model_name}')
        return np.vstack(embeddings_list)

class FeatureSelection:
    def default_selection(df, threshold=0.05):
        from sklearn.feature_selection import VarianceThreshold
        sel = VarianceThreshold(threshold=threshold)
        X_selection = sel.fit_transform(df)
        return X_selection

    def mutual_info_selection(X, y, threshold=0.0001):
        logging.info(f'Mutual Info Feature Selection. Threshold used: {threshold}')
        mutual_info = mutual_info_classif(X, y)
        mutual_info = pd.Series(mutual_info)
        mutual_info.index = X.columns
        ig_df = pd.DataFrame({
            'Feature': X.columns,
            'Information Gain': mutual_info
        }).sort_values(by='Information Gain', ascending=False)
        return ig_df[ig_df['Information Gain'] > threshold]['Feature'].values

    def filter_selection(X, y):
        selector = SelectKBest(score_func=mutual_info_classif, k=10)
        selector.fit(X, y)
        return X.columns[selector.get_support()]

    def hybrid_selection(X, y):
        logging.info(f'Hybrid method combining SelectFromModel and Logistic Regression')
        selector = SelectFromModel(LogisticRegression(penalty="l2", C=0.1))
        X_selected = selector.fit_transform(X, y)
        return X.columns[selector.get_support()]

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

# # Test model Initialize model
# model = BiLSTMClassifier(input_dim=768, hidden_dim=128, output_dim=5, num_layers=2)
# input_data = torch.randn(2, 5, 768)  # Example input (batch_size=32, seq_len=50, input_dim=768)
# output= model(input_data)
# print(output.shape)  # Expected output: (batch_size, output_dim)

def train_val_dl_models(model, train_loader, val_loader, max_grad_norm=1.0, epochs=32, lr=0.0001):
    logging.info(f'{model.__class__.__name__}; lr={lr}')
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr, momentum=0.9)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
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
        val_preds, val_labels, val_scores, val_loss = [], [], [], 0
        with torch.no_grad():  
            for inputs, labels in val_loader:
                outputs= model(inputs)
                val_loss += criterion(outputs.squeeze(), labels).item()  
                val_preds.append(torch.sigmoid(outputs))  
                val_labels.append(labels) 
                val_scores.append(outputs)
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_preds = (val_preds > 0.5).float() 
        val_accuracy = accuracy_score(val_labels.numpy(), val_preds.numpy())
        if epoch % 4 == 0:
            logging.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, 'f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')
    return val_accuracy, val_preds, torch.cat(val_scores)

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

        val_accuracy, val_preds, val_scores = train_single_fold(model, train_loader=train_loader, val_loader=val_loader, max_grad_norm=max_grad_norm, epochs=epochs, lr=lr)
        fold_results.append(val_accuracy)
        logging.info(f"Fold {fold + 1} Accuracy: {val_accuracy:.4f}")

    avg_accuracy = sum(fold_results) / k
    logging.info(f"Average Accuracy across {k} folds: {avg_accuracy:.4f}")
    return fold_results, avg_accuracy

class My_training_class:
    def __init__(self, filepath, model_list, embedding_model=None, demo=False, kFold=False):
        self.df = None
        self.filepath = filepath
        self.models = model_list if model_list is not None else ['svm', 'lr', 'rf', 'xgb', 'bilstm', 'mlp']
        self.embedding_model = embedding_model
        self.demo = True if demo == 'true' or demo == "True" or demo is True else False
        self.kFold = kFold
        self.all_outputs= {}
        for model in model_list:
            self.all_outputs[model] = {}
        self.traits = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']

    def process_raw_files(self, main_file, liwc_file, is_mypersonality=True):
        logging.info(f"Reading RAW files: {main_file} and {liwc_file}, filetype: {is_mypersonality}")
        self.df = DataProcessor.read_data(main_file, liwc_file, is_mypersonality)
        self.df = DataProcessor.process_NRC_emotion(self.df)
        self.df = DataProcessor.process_NRC_VAD(self.df)
        self.df = DataProcessor.process_VADER_sentiment(self.df)
        logging.info(f"Preprocessing RAW files completed: {self.df.shape}")

    def preprocess_data(self):
        logging.info(f'Preprocessing {self.filepath}')
        self.df = pd.read_csv(self.filepath) if self.df is None else self.df
        self.df = self.df[:2000] if self.demo else self.df
        self.df =  DataProcessor.clean_up_text(self.df)
        self.contextual_embeddings = DataProcessor.process_embeddings(self.df, self.embedding_model) if self.embedding_model else None
        logging.info(f'Preprocessing Completed. Total shape={self.df.shape}')

    def prepare_dataset(self, target_col, test_size=None):
        all_cols = self.df.columns
        remove_cols = ['STATUS', 'original', 'cEXT','cNEU', 'cAGR', 'cCON', 'cOPN']
        emb_cols = ['bert_embeddings', 'berttweet_embeddings', 'xlnet_embeddings', 'roberta_embeddings']
        stat_cols = list (set(all_cols) - set(remove_cols) - set(emb_cols))

        # selected_df = FeatureSelection.default_selection(self.df)
        selected_features = FeatureSelection.mutual_info_selection(self.df[stat_cols], self.df[target_col], 0.001)
        # selected_features = FeatureSelection.hybrid_selection(self.df[stat_cols], self.df[target_col])
        logging.info(f'selected Features: {selected_features}')
        scaler = StandardScaler() 
        stat_features_scaled = scaler.fit_transform(self.df[selected_features])
    
        X = np.concatenate([stat_features_scaled, self.contextual_embeddings] if self.contextual_embeddings is not None else [stat_features_scaled], axis=1)
        y = np.array(self.df[[target_col]]) 
        logging.info(f'statistical embedding: {stat_features_scaled.shape} ')
        logging.info(f'contextual embedding: {self.contextual_embeddings.shape if self.contextual_embeddings is not None else  []} ')
        logging.info(f'total embedding: {X.shape} ')
        if test_size is None:
            logging.info(f'Skipping split.')
            return X, y

        # print("Type of y:", type(y))
        # print("Shape of y:", y.shape)
        y = y.ravel()  # This will reshape y to (n_samples,)
        # print("After Shape of y:", y.shape)

        if self.kFold:
            logging.info(f'Skipping Data Split.')
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=test_size, random_state=42)
            # X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
            logging.info(f'Test size: {test_size}, Train  size: {self.X_train.shape}, Val size: {self.X_val.shape}')
            X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        logging.info(f'Data Preparation Completed.')

    def init_models(self):
        for model in self.models:
            if model =='svm':
                self.svm_model = SVC(kernel='linear')
            elif model == 'lr':
                self.lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
                # self.lr_model = LogisticRegression(solver='saga', max_iter=1000)
            elif model == 'rf':
                self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model == 'xgb':
                self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
            elif model == 'bilstm':   
                self.bilstm_model = BiLSTMClassifier(input_dim=self.X_train.shape[1], hidden_dim=128, output_dim=1, num_layers=2,bidirectional=True, do_attention=True, dropout_rate=0.5)
            elif model == 'mlp':  
                self.mlp_model = MLP(input_size=self.X_train.shape[1], hidden_size=128, output_size=1, dropout_rate=0.3)
        logging.info(f'Model Initiated.')
    
    def fit_and_save_models(self, target_col):
        logging.info(f'Fitting and Saving Models...')
        for model in self.models:
            if model =='svm':
                self.svm_model.fit(self.X_train, self.y_train)
                # self.svm_model.save_model(f"{ckpt}/{self.svm_model.__class__.__name__}_{target_col}.json")
            elif model == 'lr':
                self.lr_model.fit(self.X_train, self.y_train)
                # self.lr_model.save_model(f"{ckpt}/{self.lr_model.__class__.__name__}_{target_col}.json")
            elif model == 'rf':
                self.rf_model.fit(self.X_train, self.y_train)
                # self.rf_model.save_model(f"{ckpt}/{self.rf_model.__class__.__name__}_{target_col}.json")
            elif model == 'xgb': 
                self.xgb_model.fit(self.X_train, self.y_train)
                self.xgb_model.save_model(f"{ckpt}/{self.xgb_model.__class__.__name__}_{target_col}.json")
            elif model == 'bilstm':   
                self.bilstm_acc, y_pred, y_scores = train_val_dl_models(self.bilstm_model, self.train_loader, self.val_loader)
                self.all_outputs[model][target_col] = (self.y_val, y_pred, y_scores)
                torch.save(self.bilstm_model.state_dict(), f"{ckpt}/{self.bilstm_model.__class__.__name__}_{target_col}.pth")
            elif model == 'mlp':  
                self.mlp_acc, y_pred, y_scores = train_val_dl_models(self.mlp_model, self.train_loader, self.val_loader)
                self.all_outputs[model][target_col] = (self.y_val, y_pred, y_scores)
                torch.save(self.mlp_model.state_dict(), f"{ckpt}/{self.mlp_model.__class__.__name__}_{target_col}.pth")

    def validate_and_generate_acc_scr(self, target_col):
        logging.info(f'Validating Models...')
        logging.info(20*'=')
        for model in self.models:
            if model =='svm':
                svm_y_pred = self.svm_model.predict(self.X_val)
                svm_y_probs = self.svm_model.predict_proba(self.X_val)[:, 1]
                svm_accuracy = accuracy_score(self.y_val, svm_y_pred)
                logging.info(f'SVM Val Acc: {svm_accuracy:.2f}')
                self.all_outputs[model][target_col] = (self.y_val, svm_y_pred, svm_y_probs)
            elif model == 'lr':
                lr_y_pred = self.lr_model.predict(self.X_val)
                lr_y_probs = self.lr_model.predict_proba(self.X_val)[:, 1]
                lr_accuracy = accuracy_score(self.y_val, lr_y_pred)
                self.all_outputs[model][target_col] = (self.y_val, lr_y_pred, lr_y_probs)
                logging.info(f'LR Val Acc: {lr_accuracy:.2f}')
            elif model == 'rf':
                rf_y_pred = self.rf_model.predict(self.X_val)
                rf_y_proba = self.rf_model.predict_proba(self.X_val)[:, 1]
                rf_accuracy = accuracy_score(self.y_val, rf_y_pred)
                self.all_outputs[model][target_col] = (self.y_val, rf_y_pred, rf_y_proba)
                logging.info(f'RF Val Acc: {rf_accuracy:.2f}')
            elif model == 'xgb': 
                xgb_y_pred = self.xgb_model.predict(self.X_val)
                xgb_y_proba = self.xgb_model.predict_proba(self.X_val)[:, 1]
                xgb_accuracy = accuracy_score(self.y_val, xgb_y_pred)
                self.all_outputs[model][target_col] = (self.y_val, xgb_y_pred, xgb_y_proba)
                logging.info(f'SGBoost Val Acc: {xgb_accuracy:.2f}')
            elif model == 'bilstm':   
                # bilstm_acc = train_val_dl_models(self.bilstm_model, self.train_loader, self.val_loader)
                logging.info(f'BiLSTM Val Acc: {self.bilstm_acc:.2f}')
            elif model == 'mlp':  
                # mlp_acc = train_val_dl_models(self.mlp_model, self.train_loader, self.val_loader, target_col)
                logging.info(f'MLP Val Acc: { self.mlp_acc:.2f}') 
        logging.info(20*'=')

    def display_metrics(self, savefig=True):
        logging.info(f'generating metrics and confusion matrix ..')
        performance_records = [] 
        for model in self.models:
            logging.info(15*'='+f" {model} "+ 15*'=')
            a_output = self.all_outputs[model]
            n_classifiers = len(a_output)
            fig, axes = plt.subplots(1, n_classifiers, figsize=(5 * n_classifiers, 5))
            for ax, (name, (y_true, y_pred, _)) in zip(axes, a_output.items()):
                cm = confusion_matrix(y_true, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
                disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
                ax.set_title(name)

                tn, fp, fn, tp = cm.ravel()
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero
                false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # Avoid division by zero
                logging.info(10*'-'+f" {name} "+ 10*'-')
                logging.info(f"Accuracy: {accuracy:.2f}")
                logging.info(f"Precision: {precision:.2f}")
                logging.info(f"Recall (Sensitivity): {recall:.2f}")
                logging.info(f"F1-Score: {f1:.2f}")
                logging.info(f"Specificity: {specificity:.2f}")
                logging.info(f"False Positive Rate: {false_positive_rate:.2f}")
                logging.info(f"Confusion Matrix: {cm}")

                performance_records.append({
                    "Model": model,
                    "Classifier": name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                    "Specificity": specificity,
                    "False Positive Rate": false_positive_rate,
                    "Confusion Matrix": cm 
                })

            plt.tight_layout()
        if savefig:
            plt.savefig(f"{ckpt}/{model}_cm.png")
        performance_df = pd.DataFrame(performance_records)
        logging.info(f"Performance metrics dataframe created with shape: {performance_df.shape}")
        performance_df.to_csv(f"{ckpt}/performance.csv")
        for col in self.traits:
            s = performance_df[performance_df['Classifier'] ==col]
            best_model_row = s.loc[s['Accuracy'].idxmax()]
            logging.info(f'For {best_model_row["Classifier"]}, {best_model_row["Model"]},  {best_model_row["Accuracy"]}')

    def generate_auroc(self, savefig=True):      
        for model in self.models:
            plt.figure(figsize=(8, 6))
            a_output = self.all_outputs[model]
            for name, (y_true, y_pred, y_scores) in a_output.items():
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                # Compute Youden's Index
                # logging.info(f"Thresholds: {thresholds}")
                youden_index = tpr - fpr
                optimal_idx = youden_index.argmax()
                optimal_threshold = thresholds[optimal_idx]
                logging.info(f"Optimal Threshold: {optimal_threshold}")
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guessing")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for Multiple (Trait) Classifiers using {model}')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            if savefig:
                plt.savefig(f'{ckpt}/{model}_auroc.png')
            plt.show()

    #TODO
    def explain_SHAP(self, savefig=True):
        import shap
        shap.initjs()
        explainer = shap.Explainer(self.lr_model, self.X_train)
        shap_values = explainer(self.X_val)
        shap.plots.beeswarm(shap_values)
        if savefig:
            plt.savefig('shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

    def begin_training(self):
        logging.info(f'Training started with {self.embedding_model} Embedding for {self.models}') 
        # self.process_raw_files('data/mypersonality.csv', 'data/LIWC_mypersonality_oct_2.csv', True)
        self.preprocess_data()
        logging.info(70*'>')
        for target_col in self.traits:
            logging.info(10*'-')
            logging.info(f'Trait: {target_col}')
            logging.info(10*'-')
            self.prepare_dataset(target_col, test_size=0.1)
            self.init_models()
            self.fit_and_save_models(target_col)
            self.validate_and_generate_acc_scr(target_col)
            logging.info(70*'>')
        self.display_metrics()
        self.generate_auroc()

if __name__ == "__main__":
    try:
        demo = sys.argv[1]  
        emb = sys.argv[2]
        models = sys.argv[3]
        data_type = sys.argv[4]
        # kFold = sys.argv[5]
        kFold = False
        print(demo, emb, models, data_type, kFold)
        emb_models = {'1':'roberta-base', '2':'bert-base-uncased', '3':'vinai/bertweet-base', '4':'xlnet-base-cased'}
        data_types = {
            'rd': 'data/pandora_processed_train.csv', 
            'fb': 'data/mypersonality_processed_data_nov_27.csv',
            'both':'data/all_processed_train_data_nov_27.csv'
            }
        emb = emb_models[emb] if emb in emb_models.keys() else None
        models = ['lr', 'rf', 'xgb', 'mlp', 'bilstm'] if models == 'all' else ["mlp"]
        filepath = data_types[data_type] if data_type in data_types.keys() else None
        print(demo, emb, models, filepath, kFold)
        
        ## Initializing log
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ckpt = f"checkpoint/{emb.split('-')[0]}-{timestamp}_{data_type}" if emb else f"checkpoint/{timestamp}"
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        logging.basicConfig(filename=f'{ckpt}/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       
        my_train = My_training_class(filepath, model_list=models, embedding_model=emb, demo=demo, kFold=kFold)
        my_train.begin_training()
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
