import pandas as pd
import numpy as np
import re,os, glob, traceback, nltk, logging, sys
from datetime import datetime
from collections import defaultdict
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif, VarianceThreshold, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

class PreProcessor:
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
    
    def split_dataset(df, test_size=0.2):
        df_train, df_temp = train_test_split(df, test_size=test_size, shuffle=True, random_state=42)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, shuffle=True, random_state=42)
        logging.info(f'Split size: {df_train.shape}, {df_val.shape}, {df_test.shape}')
        return df_train, df_val, df_test


class FeatureSelection:
    def variance_selection(X, threshold=0.16): #(.8 * (1 - .8))
        logging.info(f'Variance Feature Selection. Threshold used: {threshold}')
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        return  X.columns[selector.get_support()]

    def mutual_info_selection(X, y, threshold=0.001):
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
        selector.fit(X, y)
        return X.columns[selector.get_support()]

    def get_optimal_features(X, y, clf=LogisticRegression(), folds=5):
        min_features_to_select = 1  # Minimum number of features to consider
        cv = StratifiedKFold(folds)
        rfecv = RFECV(
            estimator=clf,
            step=1,
            cv=cv,
            scoring="accuracy",
            min_features_to_select=min_features_to_select,
            n_jobs=2,
        )
        rfecv.fit(X, y)
        print(f"Optimal number of features: {rfecv.n_features_}")
        # X_selected = rfecv.fit_transform(X, y)
        # return X.columns[rfecv.get_support()]

if __name__ == "__main__":
    try:

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logging.basicConfig(filename=f'log/DataProcessor_log_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        main_file = "data/pandora_to_big5.csv"
        liwc_file = "data/LIWC_pandora_to_big5_oct_24.csv"
        logging.info(f"Reading RAW files: {main_file} and {liwc_file}")
        df = PreProcessor.read_data(main_file, liwc_file, False)
        df = PreProcessor.process_NRC_emotion(df)
        df = PreProcessor.process_NRC_VAD(df)
        df = PreProcessor.process_VADER_sentiment(df)
        df =  PreProcessor.clean_up_text(df)
        # df_train, df_val, df_test = PreProcessor.split_dataset(df, 0.1)
        df_train, df_test = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
        df_train.to_csv('processed_data/2-splits/pandora_train_val.csv')
        # df_val.to_csv('data/processed_data/3-splits/pandora_val.csv')
        df_test.to_csv('processed_data/2-splits/pandora_test.csv')
        logging.info(f"All files saved in process_data dir.")
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  