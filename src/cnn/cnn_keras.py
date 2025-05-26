import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import os

class CNNKeras:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        # Load CIFAR-10 data
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train_full = x_train_full.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten labels
        y_train_full = y_train_full.flatten()
        y_test = y_test.flatten()
        
        # Split training data into train and validation (4:1 ratio)
        # 40k train, 10k validation
        split_idx = 40000
        x_train = x_train_full[:split_idx]
        y_train = y_train_full[:split_idx]
        x_val = x_train_full[split_idx:]
        y_val = y_train_full[split_idx:]
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        
        print(f"Training set: {x_train.shape}")
        print(f"Validation set: {x_val.shape}")
        print(f"Test set: {x_test.shape}")
        
    def create_model(self, num_conv_layers=3, filters_per_layer=[32, 64, 128], 
                     kernel_sizes=[3, 3, 3], pooling_type='max'):
        model = keras.Sequential()
        
        # Add convolutional layers
        for i in range(num_conv_layers):
            if i == 0:
                model.add(layers.Conv2D(
                    filters_per_layer[i], 
                    kernel_sizes[i], 
                    activation='relu', 
                    input_shape=(32, 32, 3)
                ))
            else:
                model.add(layers.Conv2D(
                    filters_per_layer[i], 
                    kernel_sizes[i], 
                    activation='relu'
                ))
            
            # Add pooling layer
            if pooling_type == 'max':
                model.add(layers.MaxPooling2D((2, 2)))
            else:
                model.add(layers.AveragePooling2D((2, 2)))
        
        # Flatten and add dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, epochs=20, batch_size=32):
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
        plt.savefig(f'plots/cnn_{safe_title.lower()}.jpg', dpi=300, bbox_inches='tight', format='jpg')
        plt.show()
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def run_cnn_experiments():
    
    # Initialize CNN
    cnn = CNNKeras()
    cnn.load_and_preprocess_data()
    
    results = {}
    
    # Experiment 1: Number of convolutional layers
    print("=== Experiment 1: Number of Convolutional Layers ===")
    layer_configs = [
        (2, [32, 64], [3, 3]),
        (3, [32, 64, 128], [3, 3, 3]),
        (4, [32, 64, 128, 256], [3, 3, 3, 3])
    ]
    
    for i, (num_layers, filters, kernels) in enumerate(layer_configs):
        print(f"\nTraining model with {num_layers} conv layers...")
        cnn.create_model(num_layers, filters, kernels)
        cnn.train_model(epochs=10)
        f1, _ = cnn.evaluate_model()
        cnn.plot_training_history(f"Model with {num_layers} Conv Layers")
        cnn.save_model(f"models/cnn_layers_{num_layers}.h5")
        results[f"layers_{num_layers}"] = f1
    
    # # Experiment 2: Number of filters per layer
    # print("\n=== Experiment 2: Number of Filters per Layer ===")
    # filter_configs = [
    #     [16, 32, 64],
    #     [32, 64, 128],
    #     [64, 128, 256]
    # ]
    
    # for i, filters in enumerate(filter_configs):
    #     print(f"\nTraining model with filters {filters}...")
    #     cnn.create_model(3, filters, [3, 3, 3])
    #     cnn.train_model(epochs=10)
    #     f1, _ = cnn.evaluate_model()
    #     cnn.plot_training_history(f"Model with filters {filters}")
    #     cnn.save_model(f"models/cnn_filters_{i+1}.h5")
    #     results[f"filters_{i+1}"] = f1
    
    # # Experiment 3: Kernel sizes
    # print("\n=== Experiment 3: Kernel Sizes ===")
    # kernel_configs = [
    #     [3, 3, 3],
    #     [5, 5, 5],
    #     [7, 5, 3]
    # ]
    
    # for i, kernels in enumerate(kernel_configs):
    #     print(f"\nTraining model with kernel sizes {kernels}...")
    #     cnn.create_model(3, [32, 64, 128], kernels)
    #     cnn.train_model(epochs=10)
    #     f1, _ = cnn.evaluate_model()
    #     cnn.plot_training_history(f"Model with kernels {kernels}")
    #     cnn.save_model(f"models/cnn_kernels_{i+1}.h5")
    #     results[f"kernels_{i+1}"] = f1
    
    # # Experiment 4: Pooling types
    # print("\n=== Experiment 4: Pooling Types ===")
    # pooling_types = ['max', 'average']
    
    # for pooling in pooling_types:
    #     print(f"\nTraining model with {pooling} pooling...")
    #     cnn.create_model(3, [32, 64, 128], [3, 3, 3], pooling)
    #     cnn.train_model(epochs=10)
    #     f1, _ = cnn.evaluate_model()
    #     cnn.plot_training_history(f"Model with {pooling} pooling")
    #     cnn.save_model(f"models/cnn_pooling_{pooling}.h5")
    #     results[f"pooling_{pooling}"] = f1
    
    # Print summary
    print("\n=== EXPERIMENT RESULTS SUMMARY ===")
    for exp, f1 in results.items():
        print(f"{exp}: F1-Score = {f1:.4f}")
    
    return results

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Run experiments
    results = run_cnn_experiments()
