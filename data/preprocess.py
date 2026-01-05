import os
import zipfile
import requests
import pandas as pd
import numpy as np

def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    data_dir = "data/ml-100k"
    zip_path = "data/ml-100k.zip"

    if not os.path.exists(data_dir):
        print("MovieLens dataset downloading...")
        r = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
        os.remove(zip_path)
        print("Dataset ready")
    else:
        print("Dataset already present.")

def load_data():
    # Ratings loadings from u.data file
    path = "data/ml-100k/u.data"
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='\t', names=columns)
    
    # IDs adjustment to start from 0
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1
    
    return df

def get_train_test_split(df, test_size=0.2):
    # Shuffle the data
    df = df.sample(frac=1, random_state=42)
    split_idx = int(len(df) * (1 - test_size))
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    return train, test

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    download_movielens()
    df = load_data()
    print(f"User count: {df.user_id.nunique()}")
    print(f"Item count: {df.item_id.nunique()}")
    print(f"Rating count: {len(df)}")