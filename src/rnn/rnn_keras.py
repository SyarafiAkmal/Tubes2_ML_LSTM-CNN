import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from utils.dataset_loader import DatasetLoader
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RNNKeras:
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
        # load dataset
        df = self.dataset_loader.download_nusax_sentiment()
        
        # data splits
        X_train_text, X_val_text, X_test_text, y_train, y_val, y_test = self.dataset_loader.prepare_text_data(df)
        
        # tokenization
        self.tokenizer = layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_sequence_length=self.max_length,
            output_mode='int'
        )
        self.tokenizer.adapt(X_train_text)

        self.x_train = self.tokenizer(X_train_text).numpy()
        self.x_val = self.tokenizer(X_val_text).numpy()
        self.x_test = self.tokenizer(X_test_text).numpy()
        
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"Training set: {self.x_train.shape}")
        print(f"Validation set: {self.x_val.shape}")
        print(f"Test set: {self.x_test.shape}")
        print(f"Vocabulary size: {self.vectorizer.vocabulary_size()}")
        
    def create_model(self, num_rnn_layers=2, rnn_units=[64, 32], bidirectional=True, embedding_dim=128):
        model = keras.Sequential()
        
        # embedding layer
        model.add(layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=embedding_dim,
            input_length=self.max_length
        ))
        
        # RNN layer
        for i in range(num_rnn_layers):
            return_sequences = i < num_rnn_layers - 1
            
            if bidirectional:
                model.add(layers.Bidirectional(
                    layers.SimpleRNN(rnn_units[i], return_sequences=return_sequences)
                ))
            else:
                model.add(layers.SimpleRNN(rnn_units[i], return_sequences=return_sequences))
        
        # dropout layer
        model.add(layers.Dropout(0.5))
        
        # dense output layer
        model.add(layers.Dense(3, activation='softmax'))
        
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
        
        # save plot image
        safe_title = title.replace(" ", "_").replace("/", "_")
        plt.savefig(f'plots/rnn_{safe_title.lower()}.jpg', dpi=300, bbox_inches='tight', format='jpg')
        plt.show()
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def run_rnn_experiments():   
    rnn = RNNKeras()
    rnn.load_and_preprocess_data()
    
    results = {}
    
    # testing 1 pengaruh jumlah layer rnn
    print("=== Experiment 1: Number of RNN Layers ===")
    layer_configs = [
        (1, [64]),
        (2, [64, 32]),
        (3, [64, 64, 32])
    ]
    
    for i, (num_layers, units) in enumerate(layer_configs):
        print(f"\nTraining model with {num_layers} RNN layers...")
        rnn.create_model(num_layers, units)
        rnn.train_model(epochs=5)
        f1, _ = rnn.evaluate_model()
        rnn.plot_training_history(f"Model with {num_layers} RNN Layers")
        rnn.save_model(f"models/rnn_layers_{num_layers}.h5")
        results[f"rnn_layers_{num_layers}"] = f1
    
    # # testing 2 pengaruh banyak cell rnn tiap layer
    # print("\n=== Experiment 2: Number of RNN Units per Layer ===")
    # unit_configs = [
    #     [32, 16],
    #     [64, 32],
    #     [128, 64]
    # ]
    
    # for i, units in enumerate(unit_configs):
    #     print(f"\nTraining model with units {units}...")
    #     rnn.create_model(2, units)
    #     rnn.train_model(epochs=5)
    #     f1, _ = rnn.evaluate_model()
    #     rnn.plot_training_history(f"Model with units {units}")
    #     rnn.save_model(f"models/rnn_units_{i+1}.h5")
    #     results[f"rnn_units_{i+1}"] = f1
    
    # # testing 3 pengaruh jenis layer rnn
    # print("\n=== Experiment 3: Bidirectional vs Unidirectional ===")
    # direction_configs = [False, True]
    
    # for bidirectional in direction_configs:
    #     direction_name = "bidirectional" if bidirectional else "unidirectional"
    #     print(f"\nTraining {direction_name} model...")
    #     rnn.create_model(2, [64, 32], bidirectional)
    #     rnn.train_model(epochs=5)
    #     f1, _ = rnn.evaluate_model()
    #     rnn.plot_training_history(f"{direction_name.capitalize()} Model")
    #     rnn.save_model(f"models/rnn_{direction_name}.h5")
    #     results[f"rnn_{direction_name}"] = f1
    
    print("\n=== RNN EXPERIMENT RESULTS SUMMARY ===")
    for exp, f1 in results.items():
        print(f"{exp}: F1-Score = {f1:.4f}")
    
    return results

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    results = run_rnn_experiments()
