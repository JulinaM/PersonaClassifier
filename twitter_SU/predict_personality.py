import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc
import psutil
import warnings
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.insert(0,'/data/jmharja/projects/PersonaClassifier/')
from PersonaClassifier import My_training, Dataset
from utils.Models import MyEstimator, BiLSTMClassifier
from utils.Training import train_val_kfold, train_val, predict, train

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CHECKPOINT = '/data/jmharja/projects/PersonaClassifier/checkpoint/v4_bilstm_roberta-2025-01-22_11-14-52_final_eval_the_best_S1/models/'
OUTPUT_DIR = '/data/jmharja/projects/PersonaClassifier/twitter_SU/classification/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize GPU monitoring
try:
    nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

def print_memory_stats():
    """Print detailed memory statistics"""
    process = psutil.Process(os.getpid())
    print(f"\nCPU Memory - Used: {process.memory_info().rss / 1024 ** 2:.2f} MB | Available: {psutil.virtual_memory().available / 1024 ** 2:.2f} MB")
    
    if GPU_AVAILABLE:
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU Memory - Used: {info.used / 1024 ** 2:.2f} MB | Free: {info.free / 1024 ** 2:.2f} MB | Total: {info.total / 1024 ** 2:.2f} MB")

class MemorySafeEmbedder:
    def __init__(self, model_name='roberta-base'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = self._get_safe_device()
        
    def _get_safe_device(self):
        """Determine safest available device with fallback to CPU"""
        try:
            if torch.cuda.is_available():
                # Check GPU memory availability
                if GPU_AVAILABLE:
                    handle = nvmlDeviceGetHandleByIndex(0)
                    info = nvmlDeviceGetMemoryInfo(handle)
                    if info.free > 2 * 1024 ** 3:  # At least 2GB free
                        return torch.device('cuda:0')
            return torch.device('cpu')
        except:
            return torch.device('cpu')
    
    def load_model(self):
        """Load model with memory safety checks"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.model is None:
            try:
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("GPU memory exhausted, falling back to CPU")
                    self.device = torch.device('cpu')
                    self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                    self.model.eval()
                else:
                    raise
    
    def unload_model(self):
        """Completely unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        gc.collect()
    
    def generate_embeddings(self, texts, batch_size=16, max_length=64):
        """Generate embeddings with ultra-conservative memory usage"""
        self.load_model()
        embeddings = []
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + batch_size].tolist()
                
                with torch.no_grad():
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True
                        # max_length=max_length
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(cls_embeddings)
                    
                    # Aggressive cleanup
                    del inputs, outputs, cls_embeddings
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Memory monitoring
                if i % (100 * batch_size) == 0:
                    print_memory_stats()
                    
        except Exception as e:
            self.unload_model()
            raise e
            
        return np.vstack(embeddings)

class PersonalityPredictorCPU:
    """CPU-only predictor for when GPU fails"""
    def __init__(self, checkpoint_path, target_cols):
        self.device = torch.device('cpu')
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self.target_cols = target_cols
        self.models = {}
    
    def load_model(self, target_col):
        if target_col not in self.models:
            model =  BiLSTMClassifier(input_dim=768, hidden_dim=256, output_dim=1, num_layers=2, bidirectional=True, do_attention=True, dropout_rate=0.0001)
            model_path = f"{self.checkpoint_path}/BiLSTMClassifier_{target_col}.json"
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            # model.load_state_dict(torch.load(model_path))
            model.to(self.device)
            model.eval()
            self.models[target_col] = model
        return self.models[target_col]
    
    def predict_single_target(self, X, target_col, batch_size=512):
        model = self.load_model(target_col)
        all_probas = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i+batch_size]).float().to(self.device)
                logits = model(batch)
                probas = torch.sigmoid(logits).cpu().numpy()
                all_probas.extend(probas)
                del batch, logits
                gc.collect()
        return np.array(all_probas)


def process_year_safely(year, embedder, predictor_cpu, max_attempts=3):
    """Process a year's data with multiple recovery attempts"""
    print(f"\n{'='*50}\nProcessing year: {year}\n{'='*50}")
    
    # Load data in smallest possible chunks
    chunk_size = 500000  # Start with 50k rows per chunk
    success = False
    
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"\nAttempt {attempt}/{max_attempts} with chunk size {chunk_size}")
            chunks = pd.read_csv(f'/data2/julina/scripts/tweets/cleaned_data_by_year/{year}.csv', chunksize=chunk_size)
            
            full_results = []
            for chunk_idx, df in enumerate(chunks):
                print(f"\nProcessing chunk {chunk_idx + 1}")
                df = df.drop_duplicates(subset=['text', 'created_at'])
                df = df.loc[:, ~df.columns.str.match('Unnamed')]
                print(f"Chunk shape: {df.shape}")
                print_memory_stats()
                
                # Generate embeddings
                X = embedder.generate_embeddings(df['text'], batch_size=32, max_length=32)
                embedder.unload_model()
                
                # Make predictions
                for target_col in ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']:
                    try:
                        probas = predictor_cpu.predict_single_target(X, target_col, batch_size=256)
                        df[target_col] = probas
                    except Exception as e:
                        print(f"Error predicting {target_col}: {str(e)}")
                        df[target_col] = np.nan  # Mark as failed
                
                full_results.append(df)
                print_memory_stats()
            
            # Combine and save results
            if full_results:
                result = pd.concat(full_results, ignore_index=True)
                output_path = f'{OUTPUT_DIR}/{year}_personality.csv'
                result.to_csv(output_path, index=False)
                print(f"\nSuccessfully processed {year}. Saved to {output_path}")
                print(f"Final shape: {result.shape}")
                success = True
                break
            
        except Exception as e:
            print(f"\nAttempt {attempt} failed: {str(e)}")
            chunk_size = max(10000, chunk_size // 2)  # Halve chunk size but minimum 10k
            torch.cuda.empty_cache()
            gc.collect()
            continue
    
    if not success:
        print(f"\nFailed to process {year} after {max_attempts} attempts")

def main():
    print("Starting processing with strict memory management")
    print_memory_stats()
    
    # Initialize components
    embedder = MemorySafeEmbedder()
    predictor_cpu = PersonalityPredictorCPU(CHECKPOINT, ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU'])
    
    # Process years
    # for year in ['2019', '2020', '2021']:
    for year in ['2020']:
        process_year_safely(year, embedder, predictor_cpu)
    
    # Final cleanup
    embedder.unload_model()
    torch.cuda.empty_cache()
    gc.collect()
    print("\nProcessing complete. Memory cleaned up.")

if __name__ == "__main__":
    main()
