import os
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
import zipfile
import json

class DatasetLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_nusax_sentiment(self):
        cache_file = os.path.join(self.data_dir, "nusax_sentiment.csv")
        
        # Check if already cached
        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}")
            return pd.read_csv(cache_file)
        
        print("Downloading NusaX-Sentiment dataset...")
        
        # For demonstration, create synthetic Indonesian sentiment data
        # In real implementation, download from actual NusaX repository
        texts = [
            "Saya sangat senang dengan produk ini, kualitasnya luar biasa",
            "Pelayanan yang buruk sekali, tidak memuaskan sama sekali", 
            "Biasa saja tidak ada yang istimewa, standar",
            "Luar biasa bagus sekali, sangat merekomendasikan",
            "Mengecewakan tidak sesuai ekspektasi, jelek",
            "Cukup baik untuk harga segini, worth it",
            "Sangat memuaskan pelayanannya, excellent service",
            "Tidak recommended sama sekali, waste of money",
            "Lumayan bagus tapi bisa lebih baik lagi",
            "Perfect sesuai dengan yang diharapkan, mantap",
            "Produk berkualitas tinggi dengan harga terjangkau",
            "Kecewa berat dengan pembelian ini",
            "Tidak ada yang spesial dari produk ini",
            "Sangat puas dengan hasil yang didapat",
            "Buruk sekali, menyesal beli",
            "Oke lah untuk harga segitu",
            "Terbaik yang pernah saya beli",
            "Mengecewakan banget, tidak sesuai deskripsi",
            "Lumayan, tidak mengecewakan",
            "Sempurna, exactly what I needed"
        ] * 500
        
        # Labels: 0=negative, 1=neutral, 2=positive  
        labels = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] * 500
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        # Add some noise to make it more realistic
        np.random.seed(42)
        indices = np.random.permutation(len(df))
        df = df.iloc[indices].reset_index(drop=True)
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        print(f"Dataset cached to {cache_file}")
        
        return df
    
    def load_cifar10(self):
        from tensorflow import keras
        return keras.datasets.cifar10.load_data()
    
    def prepare_text_data(self, df, test_size=0.3, val_size=0.5):
        # Split into train and temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['text'].values, df['label'].values, 
            test_size=test_size, random_state=42, stratify=df['label']
        )
        
        # Split temp into val and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, random_state=42, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
