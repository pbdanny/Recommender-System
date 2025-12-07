import pickle
import pandas as pd
import yaml
from pathlib import Path
# from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

config_path = Path(__file__).parent.parent.parent / 'params.yaml'
data_raw_path = Path(__file__).parent.parent.parent / 'data' / 'raw'
data_processed_path = Path(__file__).parent.parent.parent / 'data' / 'processed'

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
    
    # Basic preprocessing
    # 
    
    # Split data
    trainset, testset = train_test_split(data, train_size=params["train_size"], random_state=params["random_state"])
    
    # Save processed data
    data_processed_path.mkdir(parents=True, exist_ok=True)
    
    pickle.dump(trainset, open(data_processed_path/'trainset.pkl', 'wb'))
    pickle.dump(testset, open(data_processed_path/'testset.pkl', 'wb'))

if __name__ == "__main__":
    preprocess_data()