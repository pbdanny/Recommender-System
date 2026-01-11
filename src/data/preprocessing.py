import pickle
import pandas as pd
import yaml
from pathlib import Path
# from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# --- 1. PATHS ---
# Change to use current working directory
# current_dir = Path(__file__).parent
# config_path = current_dir.parent.parent / 'params.yaml'
# data_processed_dir = current_dir.parent.parent / 'data' / 'processed'
# model_path = current_dir.parent.parent / 'models'

root_dir = Path.cwd()
config_path = root_dir / 'params.yaml'
data_raw_path = root_dir / 'data' / 'raw'
data_processed_dir = root_dir / 'data' / 'processed'

def load_params():
    """Load parameters from params.yaml"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_data():
    """Preprocess raw data and split into train/test"""
    params = load_params()['preprocess']
    
    # Load raw data
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(data_raw_path/'u.data', reader=reader)
    raw_ratings = data.raw_ratings
    
    # Split data (https://surprise.readthedocs.io/en/stable/FAQ.html)
    # Train & Tune = 90% of the data, Test = 10% of the data
    threshold = int(0.9 * len(raw_ratings))
    train_raw_ratings = raw_ratings[:threshold]
    test_raw_ratings = raw_ratings[threshold:]
    
    # Save processed data
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_processed_dir/'train_data.data', 'w') as f:
        for line in train_raw_ratings:
            f.write('\t'.join(str(s) for s in line) + '\n')

    with open(data_processed_dir/'test_data.data', 'w') as f:
        for line in test_raw_ratings:
            f.write('\t'.join(str(s) for s in line) + '\n')
    
    with open(data_processed_dir/'all_data.data', 'w') as f:
        for line in raw_ratings:
            f.write('\t'.join(str(s) for s in line) + '\n')

    # pickle.dump(train_raw_ratings, open(data_processed_dir/'train_data.pkl', 'wb'))
    # pickle.dump(test_raw_ratings, open(data_processed_dir/'test_data.pkl', 'wb'))

if __name__ == "__main__":
    preprocess_data()