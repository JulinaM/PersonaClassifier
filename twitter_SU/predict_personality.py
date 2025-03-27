import pandas as pd
import numpy as np
import torch.nn as nn
import torch, os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.insert(0,'/data/jmharja/projects/PersonaClassifier/')
from PersonaClassifier import My_training, Dataset
from utils.Models import MyEstimator, BiLSTMClassifier
from utils.Training import train_val_kfold, train_val, predict, train
checkpoint = '/data/jmharja/projects/PersonaClassifier/checkpoint/v4_bilstm_roberta-2025-01-22_11-14-52_final_eval_the_best_S1/models/'


from transformers import AutoTokenizer, AutoModel
def process_embeddings(df, model_name, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()  
    embeddings_list = []
    
    for i in range(0, len(df), batch_size):
        batch_texts = df['text'][i:i + batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings_list.append(cls_embeddings.cpu().numpy())
    return np.vstack(embeddings_list)


df = pd.read_csv('/data2/julina/scripts/tweets/cleaned_data_by_year/2019.csv')
df.drop_duplicates(subset=['text', 'created_at'], inplace=True)
df = df.loc[:, ~df.columns.str.match('Unnamed')]

df_r = pd.read_csv('/data2/julina/scripts/tweets/cleaned_data_by_year/2019_race.csv')
df_r.drop_duplicates(subset=['text', 'created_at'], inplace=True)
df_r = df_r.loc[:, ~df_r.columns.str.match('Unnamed')]

df_2019 = pd.merge(df, df_r[['id', 'user_id', 'race']],  how='left', on=['id','user_id'])

print(df_2019.shape)
# df= df.drop_duplicates(subset='posts', keep='first')
# df['posts'] = df['posts'].astype(str).fillna('')
# df = df[df['posts'].str.strip() != '']
# df = df[df['posts'].str.split().str.len() >= 3]
X = process_embeddings(df_2019, 'text', 'roberta-base')
result = df
for target_col in ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU'] :
    model_filepath = f"{checkpoint}/BiLSTMClassifier_{target_col}.json"
    model =  BiLSTMClassifier(input_dim=768, hidden_dim=256, output_dim=1, num_layers=2, bidirectional=True, do_attention=True, dropout_rate=0.0001)
    model.load_state_dict(torch.load(model_filepath))
    print(f"Model loaded from {model_filepath}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    preds, probas = predict(model, X, device, 0.5)
    probs_df = pd.DataFrame(probas, columns=[target_col])
    result = pd.concat([result, probs_df], axis=1)
result.to_csv(f'/data/jmharja/projects/PersonaClassifier/reddit_teenagers/regression/{filename}')
