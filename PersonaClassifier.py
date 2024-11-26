import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re,os, glob, traceback, nltk
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import logging, sys
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ckpt = f"checkpoint/{timestamp}"
if not os.path.exists(ckpt):
    os.makedirs(ckpt)
logging.basicConfig(filename=f'{ckpt}/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate) 
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
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

def train_val_dl_models(model, train_loader, val_loader, target_col, max_grad_norm=1.0, epochs=16, lr=0.001):
    logging.info(f'{model.__class__.__name__}')
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
            total_loss += loss.item()
        model.eval()  
        val_preds, val_labels, val_loss = [], [], 0
        with torch.no_grad():  
            for inputs, labels in val_loader:
                outputs= model(inputs)
                val_loss += criterion(outputs.squeeze(), labels).item()  
                val_preds.append(torch.sigmoid(outputs))  
                val_labels.append(labels) 
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_preds = (val_preds > 0.5).float() 
        val_accuracy = accuracy_score(val_labels.numpy(), val_preds.numpy())
        if epoch % 4 == 0:
            logging.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, 'f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')
    return val_accuracy

class My_training_class:
    def __init__(self,  model_list=['svm', 'lr', 'rf', 'xgb', 'bilstm', 'mlp'], embedding_model=None, demo=False):
        self.df = None
        self.models = model_list
        self.embedding_model = embedding_model
        self.demo = True if demo == 'true' or demo == "True" or demo is True else False

    def preprocess_data(self, main_file, liwc_file, is_mypersonality=True, is_preprocessed=False):
        if self.df is None:
            if not is_preprocessed:
                self.df = read_data(main_file, liwc_file, is_mypersonality)
                self.df = process_NRC_emotion(self.df)
                self.df = process_NRC_VAD(self.df)
                self.df = process_VADER_sentiment(self.df)
            else:
                self.df = pd.read_csv(main_file)
                logging.info(10*' Dr. Julina Maharjan ')
            if self.demo:
                self.df = self.df[:2000]
                logging.info(5*' DEMO ')
            self.contextual_embeddings = process_embeddings(self.df, self.embedding_model) if self.embedding_model else None
            self.df.fillna(value=0, inplace=True)
            self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed:')]
            logging.info(f'Preprocessing Completed. Total shape={self.df.shape}')
        else:
            logging.info(f'Skipping Preprocessing. Total shape={self.df.shape}')

    def prepare_dataset(self, target_col, test_size=0.1):
        all_cols = self.df.columns
        remove_cols = ['STATUS', 'original', 'cEXT','cNEU', 'cAGR', 'cCON', 'cOPN']
        emb_cols = ['bert_embeddings', 'berttweet_embeddings', 'xlnet_embeddings', 'roberta_embeddings']
        stat_cols = list (set(all_cols) - set(remove_cols) - set(emb_cols))
        scaler = StandardScaler() 
        stat_features = self.df[stat_cols]
        stat_features_scaled = scaler.fit_transform(stat_features)
    
        X = np.concatenate([stat_features_scaled, self.contextual_embeddings] if self.contextual_embeddings is not None else [stat_features_scaled], axis=1)
        y = np.array(self.df[[target_col]]) 
        logging.info(f'statistical embedding: {stat_features_scaled.shape} ')
        logging.info(f'contextual embedding: {self.contextual_embeddings.shape if self.contextual_embeddings is not None else  []} ')
        logging.info(f'total embedding: {X.shape} ')

        # print("Type of y:", type(y))
        # print("Shape of y:", y.shape)
        y = y.ravel()  # This will reshape y to (n_samples,)
        # print("After Shape of y:", y.shape)

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
                self.svm_model.save_model(f"{ckpt}/{self.svm_model.__class__.__name__}_{target_col}.json")
            elif model == 'lr':
                self.lr_model.fit(self.X_train, self.y_train)
                self.lr_model.save_model(f"{ckpt}/{self.lr_model.__class__.__name__}_{target_col}.json")
            elif model == 'rf':
                self.rf_model.fit(self.X_train, self.y_train)
                self.rf_model.save_model(f"{ckpt}/{self.rf_model.__class__.__name__}_{target_col}.json")
            elif model == 'xgb': 
                self.xgb_model.fit(self.X_train, self.y_train)
                self.xgb_model.save_model(f"{ckpt}/{self.xgb_model.__class__.__name__}_{target_col}.json")
            elif model == 'bilstm':   
                self.bilstm_acc = train_val_dl_models(self.bilstm_model, self.train_loader, self.val_loader)
                torch.save(self.bilstm_model.state_dict(), f"{ckpt}/{self.bilstm_model.__class__.__name__}_{target_col}.pth")
            elif model == 'mlp':  
                self.mlp_acc = train_val_dl_models(self.mlp_model, self.train_loader, self.val_loader, target_col)
                torch.save(self.mlp_model.state_dict(), f"{ckpt}/{self.mlp_model.__class__.__name__}_{target_col}.pth")


    def validate_and_generate_acc_scr(self, target_col):
        logging.info(f'Validating Models...')
        logging.info(20*'=')
        for model in self.models:
            if model =='svm':
                svm_y_pred = self.svm_model.predict(self.X_val)
                svm_accuracy = accuracy_score(self.y_val, svm_y_pred)
                logging.info(f'SVM Val Acc: {svm_accuracy:.2f}')
            elif model == 'lr':
                lr_y_pred = self.lr_model.predict(self.X_val)
                lr_accuracy = accuracy_score(self.y_val, lr_y_pred)
                logging.info(f'LR Val Acc: {lr_accuracy:.2f}')
            elif model == 'rf':
                rf_y_pred = self.rf_model.predict(self.X_val)
                rf_accuracy = accuracy_score(self.y_val, rf_y_pred)
                logging.info(f'RF Val Acc: {rf_accuracy:.2f}')
            elif model == 'xgb': 
                xgb_y_pred = self.xgb_model.predict(self.X_val)
                xgb_accuracy = accuracy_score(self.y_val, xgb_y_pred)
                logging.info(f'SGBoost Val Acc: {xgb_accuracy:.2f}')
            elif model == 'bilstm':   
                # bilstm_acc = train_val_dl_models(self.bilstm_model, self.train_loader, self.val_loader)
                logging.info(f'BiLSTM Val Acc: {self.bilstm_acc:.2f}')
            elif model == 'mlp':  
                # mlp_acc = train_val_dl_models(self.mlp_model, self.train_loader, self.val_loader, target_col)
                logging.info(f'MLP Val Acc: { self.mlp_acc:.2f}') 
        logging.info(20*'=')

    def begin_training(self):
        logging.info(f'Training started with {self.embedding_model} Embedding for {self.models}') #TODO preprocess only one time
        self.preprocess_data('data/pandora_processed_train_nov26.csv', None, False, True)
        # self.preprocess_data('data/mypersonality.csv', 'data/LIWC_mypersonality_oct_2.csv', True)
        logging.info(70*'>')
        for target_col in ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']:
            logging.info(10*'-')
            logging.info(f'Trait: {target_col}')
            logging.info(10*'-')
            self.prepare_dataset(target_col, test_size=0.1)
            self.init_models()
            self.fit_and_save_models(target_col)
            self.validate_and_generate_acc_scr(target_col)
            logging.info(70*'>')

if __name__ == "__main__":
    try:
        my_train = My_training_class(model_list=['mlp'], embedding_model='roberta-base', demo=sys.argv[1])
        my_train.begin_training()
        # my_train_bert = My_training_class(model_list=['xgb', 'mlp'], embedding_model='bert-base-uncased')
        # my_train_bert.begin_training(demo)
        # train_bert_tweet = My_training_class(model_list=['xgb', 'mlp'], embedding_model='vinai/bertweet-base')
        # train_bert_tweet.begin_training(demo)
        # my_train_xlnet = My_training_class(model_list=['xgb', 'mlp'], embedding_model='xlnet-base-cased')
        # my_train_xlnet.begin_training(demo)
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
        

