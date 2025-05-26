import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset_loader import DatasetLoader

class LSTMKeras:
    def __init__(self):
        self.model = None
        self.history = None
        self.tokenizer = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.max_length = 100
        self.vocab_size = 10000
        self.dataset_loader = DatasetLoader()
        
    def load_and_preprocess_data(self):
        # Load dataset using dataset loader
        df = self.dataset_loader.download_nusax_sentiment()
        
        # Prepare data splits
        X_train_text, X_val_text, X_test_text, y_train, y_val, y_test = \
            self.dataset_loader.prepare_text_data(df)
        
        # Create and fit tokenizer
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>"
        )
        self.tokenizer.fit_on_texts(X_train_text)
        
        # Convert texts to sequences
        x_train = self.tokenizer.texts_to_sequences(X_train_text)
        x_val = self.tokenizer.texts_to_sequences(X_val_text)
        x_test = self.tokenizer.texts_to_sequences(X_test_text)
        
        # Pad sequences
        x_train = keras.preprocessing.sequence.pad_sequences(
            x_train, maxlen=self.max_length, padding='post'
        )
        x_val = keras.preprocessing.sequence.pad_sequences(
            x_val, maxlen=self.max_length, padding='post'
        )
        x_test = keras.preprocessing.sequence.pad_sequences(
            x_test, maxlen=self.max_length, padding='post'
        )
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        
        print(f"Training set: {x_train.shape}")
        print(f"Validation set: {x_val.shape}")
        print(f"Test set: {x_test.shape}")
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        
    def create_model(self, num_lstm_layers=2, lstm_units=[64, 32], 
                     bidirectional=True, embedding_dim=128):
        model = keras.Sequential()
        
        # Embedding layer
        model.add(layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=embedding_dim,
            input_length=self.max_length
        ))
        
        # LSTM layers
        for i in range(num_lstm_layers):
            return_sequences = i < num_lstm_layers - 1  # Return sequences for all but last layer
            
            if bidirectional:
                model.add(layers.Bidirectional(
                    layers.LSTM(lstm_units[i], return_sequences=return_sequences)
                ))
            else:
                model.add(layers.LSTM(lstm_units[i], return_sequences=return_sequences))
        
        # Dropout layer
        model.add(layers.Dropout(0.5))
        
        # Dense output layer
        model.add(layers.Dense(3, activation='softmax'))  # 3 classes: negative, neutral, positive
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, epochs=10, batch_size=32):
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate macro F1-score
        f1_macro = f1_score(self.y_test, y_pred_classes, average='macro')
        
        print(f"Macro F1-Score: {f1_macro:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_classes))
        
        return f1_macro, y_pred_classes
    
    def plot_training_history(self, title="Training History"):
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{title} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title(f'{title} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save as JPG
        safe_title = title.replace(" ", "_").replace("/", "_")
        plt.savefig(f'plots/lstm_{safe_title.lower()}.jpg', dpi=300, bbox_inches='tight', format='jpg')
        plt.show()
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def run_lstm_experiments():
    
    # Initialize LSTM
    lstm = LSTMKeras()
    lstm.load_and_preprocess_data()
    
    results = {}
    
    # Experiment 1: Number of LSTM layers
    print("=== Experiment 1: Number of LSTM Layers ===")
    layer_configs = [
        (1, [64]),
        (2, [64, 32]),
        (3, [64, 64, 32])
    ]
    
    for i, (num_layers, units) in enumerate(layer_configs):
        print(f"\nTraining model with {num_layers} LSTM layers...")
        lstm.create_model(num_layers, units)
        lstm.train_model(epochs=5)
        f1, _ = lstm.evaluate_model()
        lstm.plot_training_history(f"Model with {num_layers} LSTM Layers")
        lstm.save_model(f"models/lstm_layers_{num_layers}.h5")
        results[f"lstm_layers_{num_layers}"] = f1
    
    # # Experiment 2: Number of LSTM units per layer
    # print("\n=== Experiment 2: Number of LSTM Units per Layer ===")
    # unit_configs = [
    #     [32, 16],
    #     [64, 32],
    #     [128, 64]
    # ]
    
    # for i, units in enumerate(unit_configs):
    #     print(f"\nTraining model with units {units}...")
    #     lstm.create_model(2, units)
    #     lstm.train_model(epochs=5)
    #     f1, _ = lstm.evaluate_model()
    #     lstm.plot_training_history(f"Model with units {units}")
    #     lstm.save_model(f"models/lstm_units_{i+1}.h5")
    #     results[f"lstm_units_{i+1}"] = f1
    
    # # Experiment 3: Bidirectional vs Unidirectional
    # print("\n=== Experiment 3: Bidirectional vs Unidirectional ===")
    # direction_configs = [False, True]
    
    # for bidirectional in direction_configs:
    #     direction_name = "bidirectional" if bidirectional else "unidirectional"
    #     print(f"\nTraining {direction_name} model...")
    #     lstm.create_model(2, [64, 32], bidirectional)
    #     lstm.train_model(epochs=5)
    #     f1, _ = lstm.evaluate_model()
    #     lstm.plot_training_history(f"{direction_name.capitalize()} Model")
    #     lstm.save_model(f"models/lstm_{direction_name}.h5")
    #     results[f"lstm_{direction_name}"] = f1
    
    # Print summary
    print("\n=== LSTM EXPERIMENT RESULTS SUMMARY ===")
    for exp, f1 in results.items():
        print(f"{exp}: F1-Score = {f1:.4f}")
    
    return results

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Run experiments
    results = run_lstm_experiments()
