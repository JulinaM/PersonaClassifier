import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re,os, glob, traceback, nltk
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
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
logging.basicConfig(filename=f'log/log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

my_personality_file = 'data/mypersonality.csv'
main_df = pd.read_csv(my_personality_file, encoding='Windows-1252')
main_df.drop(columns=['#AUTHID', 'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN', 'DATE', 'NETWORKSIZE', 'BETWEENNESS', 'NBETWEENNESS', 'DENSITY', 'BROKERAGE', 'NBROKERAGE','TRANSITIVITY'], inplace=True)
main_df[['cOPN', 'cEXT', 'cNEU', 'cAGR', 'cCON']] = main_df[['cOPN', 'cEXT', 'cNEU', 'cAGR', 'cCON']].replace({'y': 1, 'n': 0})
logging.info(f'File={my_personality_file} shape={main_df.shape}')

liwc_file = 'data/LIWC_mypersonality_oct_2.csv'
liwc_df = pd.read_csv(liwc_file)
liwc_df = liwc_df.drop(['Unnamed: 0', 'ColumnID', 'Text'], axis=1)
logging.info(f'File={liwc_file} shape={liwc_df.shape}')

df = pd.concat([main_df, liwc_df], axis=1)
df = df.drop(['#AUTHID'], axis=1)

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

from transformers import AutoTokenizer, AutoModel
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_embeddings(df, model_name, batch_size=8):
    logging.info(f'Embeding : {model_name}')
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
    return np.vstack(embeddings_list)
    
bert_embeddings = get_embeddings(df, 'bert-base-uncased', batch_size=2)
roberta_embeddings = get_embeddings(df, 'roberta-base', batch_size=2)
# berttweet_embeddings = get_embeddings(df, 'vinai/bertweet-base', batch_size=2)
# xlnet_embeddings = get_embeddings(df, 'xlnet-base-cased', batch_size=2)
df['bert_embeddings'] = list(bert_embeddings)
df['roberta_embeddings'] = list(roberta_embeddings)
# df['berttweet_embeddings'] = list(berttweet_embeddings)
# df['xlnet_embeddings'] = list(xlnet_embeddings)

dff = df.copy()
dff.fillna(value=0, inplace=True)
logging.info(f'Preprocessing Completed. Total shape={dff.shape}')

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

def train_val_dl_models(model, train_loader, val_loader, max_grad_norm=1.0, epochs=16, lr=0.001):
    logging.info(f'{model.__class__.__name__}')
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for t, labels in train_loader:
            optimizer.zero_grad()  
            outputs= model(t)
            loss = criterion(outputs, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()  
        val_preds, val_labels, val_loss = [], [], 0
        with torch.no_grad():  
            for inputs, labels in val_loader:
                outputs= model(inputs)
                val_loss += criterion(outputs, labels).item()  
                val_preds.append(torch.sigmoid(outputs))  
                val_labels.append(labels) 
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_preds = (val_preds > 0.5).float() 
        val_accuracy = accuracy_score(val_labels.numpy(), val_preds.numpy())
        if epochs % 4 == 0:
            logging.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, 'f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')
    return val_accuracy

class My_training_class:
    def __init__(self, dff):
        self.dff = dff
    
    def prepare_dataset(self, embedding_type, target_col):
        all_cols = self.dff.columns
        remove_cols = ['STATUS', 'cEXT','cNEU', 'cAGR', 'cCON', 'cOPN']
        emb_cols = ['bert_embeddings', 'berttweet_embeddings', 'xlnet_embeddings', 'roberta_embeddings']
        stat_cols = list (set(all_cols) - set(remove_cols) - set(emb_cols))
        if embedding_type:
            contextual_embeddings = np.array(self.dff[embedding_type].tolist())
        scaler = StandardScaler() 
        stat_features = self.dff[stat_cols]
        stat_features_scaled = scaler.fit_transform(stat_features)
        X = np.concatenate([stat_features_scaled, contextual_embeddings] if embedding_type else [stat_features_scaled], axis=1)
        y = np.array(self.dff[[target_col]]) 
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        self.input_dim = self.X_train.shape[1]
        # X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
        # logging.info(f'Train  size: {X_train.shape}, Val size: {X_val.shape}, Test  size: {X_test.shape}')
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    def init_models(self):
        self.svm_model = SVC(kernel='linear')
        self.lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        self.bilstm_model = BiLSTMClassifier(input_dim=self.input_dim, hidden_dim=128, output_dim=1, num_layers=2, bidirectional=True, do_attention=True, dropout_rate=0.5)
        self.mlp_model = MLP(input_size=self.input_dim, hidden_size=128, output_size=1, dropout_rate=0.3)
    
    def fit_validate_and_generate_acc_scr(self):
        self.svm_model.fit(self.X_train, self.y_train)
        self.lr_model.fit(self.X_train, self.y_train)
        self.rf_model.fit(self.X_train, self.y_train)
        self.xgb_model.fit(self.X_train, self.y_train)
        mlp_acc = train_val_dl_models(self.mlp_model, self.train_loader, self.val_loader)
        bilstm_acc = train_val_dl_models(self.bilstm_model, self.train_loader, self.val_loader)

        svm_y_pred = self.svm_model.predict(self.X_val)
        lr_y_pred = self.lr_model.predict(self.X_val)
        rf_y_pred = self.rf_model.predict(self.X_val)
        xgb_y_pred = self.xgb_model.predict(self.X_val)
        svm_accuracy = accuracy_score(self.y_val, svm_y_pred)
        lr_accuracy = accuracy_score(self.y_val, lr_y_pred)
        rf_accuracy = accuracy_score(self.y_val, rf_y_pred)
        xgb_accuracy = accuracy_score(self.y_val, xgb_y_pred)
        logging.info(f'SVM Val Acc: {svm_accuracy:.2f}')
        logging.info(f'LR Val Acc: {lr_accuracy:.2f}')
        logging.info(f'RF Val Acc: {rf_accuracy:.2f}')
        logging.info(f'SGBoost Val Acc: {xgb_accuracy:.2f}')
        logging.info(f'MLP Val Acc: {mlp_acc:.2f}')
        logging.info(f'BiLSTM Val Acc: {bilstm_acc:.2f}')

    # def test_model():
    #     model.eval()
    #     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    #     with torch.no_grad():  
    #         outputs = model(X_test_tensor)
    #     outputs = torch.sigmoid(outputs)
    #     preds = (outputs > 0.5).float()
    #     preds_np = preds.numpy()
    #     y_test_np = y_test  # If y_test is already in numpy format, otherwise y_test.numpy()
    #     accuracy = accuracy_score(y_test_np, preds_np)
    #     precision = precision_score(y_test_np, preds_np)
    #     recall = recall_score(y_test_np, preds_np)
    #     f1 = f1_score(y_test_np, preds_np)
            # Print metrics
            # print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    def train_all_models(self,embedding_type):
        logging.info(f'Training started ::')
        for target_cols in ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']:
            logging.info(f'Trait: {target_cols}')
            self.prepare_dataset(embedding_type, target_cols)
            self.init_models()
            self.fit_validate_and_generate_acc_scr()


my_train = My_training_class(dff)
my_train.train_all_models(None)
my_train.train_all_models('berttweet_embeddings')
my_train.train_all_models('roberta_embeddings')

