import os
import pandas as pd

class DatasetLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_nusax_sentiment(self):
        train_file = os.path.join(self.data_dir, "train.csv")
        test_file = os.path.join(self.data_dir, "test.csv")
        valid_file = os.path.join(self.data_dir, "valid.csv")
        
        required_files = [train_file, test_file, valid_file]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required dataset files: {missing_files}")
        
        print("Loading dataset from CSV files...")
        
        # load CSV files
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        valid_df = pd.read_csv(valid_file)
        
        # validate columns
        required_columns = ['id', 'text', 'label']
        for df_name, df in [('train.csv', train_df), ('test.csv', test_df), ('valid.csv', valid_df)]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{df_name} is missing required columns: {missing_cols}")
        
        # convert text labels to numeric labels
        # positive -> 2, neutral -> 1, negative -> 0
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        def convert_labels(df, df_name):
            df['label'] = df['label'].str.lower().str.strip()
            
            # check for unknown labels
            unknown_labels = set(df['label'].unique()) - set(label_mapping.keys())
            if unknown_labels:
                print(f"Warning: Unknown labels in {df_name}: {unknown_labels}")
                # leep only rows with known labels
                df = df[df['label'].isin(label_mapping.keys())]
            
            df['label'] = df['label'].map(label_mapping)
            return df
        
        train_df = convert_labels(train_df, 'train.csv')
        test_df = convert_labels(test_df, 'test.csv')  
        valid_df = convert_labels(valid_df, 'valid.csv')
        
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        valid_df['split'] = 'valid'
        
        combined_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)
        
        return combined_df
    
    def load_cifar10(self):
        from tensorflow import keras
        return keras.datasets.cifar10.load_data()
    
    def prepare_text_data(self, df):
        # extract data based on the 'split' column
        train_data = df[df['split'] == 'train']
        valid_data = df[df['split'] == 'valid']
        test_data = df[df['split'] == 'test']
        
        X_train = train_data['text'].values
        y_train = train_data['label'].values
        
        X_val = valid_data['text'].values
        y_val = valid_data['label'].values
        
        X_test = test_data['text'].values
        y_test = test_data['label'].values
        
        return X_train, X_val, X_test, y_train, y_val, y_test