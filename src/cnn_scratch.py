import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ========== CNN FROM SCRATCH IMPLEMENTATION ==========
class Conv2DLayer:
    def __init__(self, weights, bias, activation='relu', padding='same'):
        self.weights = weights  # Shape: (kernel_h, kernel_w, input_channels, output_channels)
        self.bias = bias        # Shape: (output_channels,)
        self.activation = activation
        self.padding = padding
    
    def forward(self, x):
        batch_size, input_h, input_w, input_c = x.shape
        kernel_h, kernel_w, _, output_c = self.weights.shape
        
        # Handle padding
        if self.padding == 'same':
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
            output_h = input_h
            output_w = input_w
        else:  # 'valid'
            x_padded = x
            output_h = input_h - kernel_h + 1
            output_w = input_w - kernel_w + 1
        
        # Safety check
        if output_h <= 0 or output_w <= 0:
            raise ValueError(f"Invalid output dimensions: {output_h}x{output_w}. Check kernel size and padding.")
        
        # Initialize output
        output = np.zeros((batch_size, output_h, output_w, output_c))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(output_c):
                for oh in range(output_h):
                    for ow in range(output_w):
                        # Extract patch
                        patch = x_padded[b, oh:oh+kernel_h, ow:ow+kernel_w, :]
                        # Ensure patch has correct shape
                        if patch.shape != (kernel_h, kernel_w, input_c):
                            raise ValueError(f"Patch shape mismatch: expected {(kernel_h, kernel_w, input_c)}, got {patch.shape}")
                        # Compute convolution
                        output[b, oh, ow, oc] = np.sum(patch * self.weights[:, :, :, oc]) + self.bias[oc]
        
        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
        
        return output

class MaxPooling2DLayer:
    def __init__(self, pool_size=(2, 2)):
        self.pool_size = pool_size
    
    def forward(self, x):
        batch_size, input_h, input_w, channels = x.shape
        pool_h, pool_w = self.pool_size
        
        # Calculate output dimensions
        output_h = input_h // pool_h
        output_w = input_w // pool_w
        
        # Safety check
        if output_h <= 0 or output_w <= 0:
            raise ValueError(f"Invalid pooling output dimensions: {output_h}x{output_w}. Input: {input_h}x{input_w}, Pool: {pool_h}x{pool_w}")
        
        output = np.zeros((batch_size, output_h, output_w, channels))
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(output_h):
                    for ow in range(output_w):
                        h_start = oh * pool_h
                        h_end = h_start + pool_h
                        w_start = ow * pool_w
                        w_end = w_start + pool_w
                        
                        # Safety check for bounds
                        if h_end > input_h or w_end > input_w:
                            raise ValueError(f"Pooling bounds exceed input: ({h_start}:{h_end}, {w_start}:{w_end}) vs ({input_h}, {input_w})")
                        
                        pool_region = x[b, h_start:h_end, w_start:w_end, c]
                        output[b, oh, ow, c] = np.max(pool_region)
        
        return output

class AveragePooling2DLayer:
    def __init__(self, pool_size=(2, 2)):
        self.pool_size = pool_size
    
    def forward(self, x):
        batch_size, input_h, input_w, channels = x.shape
        pool_h, pool_w = self.pool_size
        
        # Calculate output dimensions
        output_h = input_h // pool_h
        output_w = input_w // pool_w
        
        # Safety check
        if output_h <= 0 or output_w <= 0:
            raise ValueError(f"Invalid pooling output dimensions: {output_h}x{output_w}. Input: {input_h}x{input_w}, Pool: {pool_h}x{pool_w}")
        
        output = np.zeros((batch_size, output_h, output_w, channels))
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(output_h):
                    for ow in range(output_w):
                        h_start = oh * pool_h
                        h_end = h_start + pool_h
                        w_start = ow * pool_w
                        w_end = w_start + pool_w
                        
                        # Safety check for bounds
                        if h_end > input_h or w_end > input_w:
                            raise ValueError(f"Pooling bounds exceed input: ({h_start}:{h_end}, {w_start}:{w_end}) vs ({input_h}, {input_w})")
                        
                        pool_region = x[b, h_start:h_end, w_start:w_end, c]
                        output[b, oh, ow, c] = np.mean(pool_region)
        
        return output

class FlattenLayer:
    def forward(self, x):
        batch_size = x.shape[0]
        flattened_size = np.prod(x.shape[1:])  # Calculate total size excluding batch dimension
        
        # Safety check
        if flattened_size == 0:
            raise ValueError(f"Cannot flatten tensor with zero size: input shape {x.shape}")
        
        result = x.reshape(batch_size, -1)
        
        # Additional safety check
        if result.shape[1] == 0:
            raise ValueError(f"Flattening resulted in zero features: {x.shape} → {result.shape}")
        
        return result

class DenseLayer:
    def __init__(self, weights, bias, activation=None):
        self.weights = weights  # Shape: (input_size, output_size)
        self.bias = bias        # Shape: (output_size,)
        self.activation = activation
    
    def forward(self, x):
        # Safety checks
        if x.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Input features ({x.shape[1]}) don't match weight input size ({self.weights.shape[0]})")
        
        output = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'softmax':
            # Stable softmax
            exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
            output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return output

class CNNFromScratch:
    def __init__(self, model=None):
        self.layers = []
        self.model = model
    
    def load_keras_model(self, model):
        """Load weights from a trained Keras model"""
        if isinstance(model, str):
            keras_model = keras.models.load_model(model)
        else:
            keras_model = model
        self.layers = []
        
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights, bias = layer.get_weights()
                activation = layer.activation.__name__ if layer.activation else None
                padding = layer.padding
                self.layers.append(Conv2DLayer(weights, bias, activation, padding))
            
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                pool_size = layer.pool_size
                self.layers.append(MaxPooling2DLayer(pool_size))
            
            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                pool_size = layer.pool_size
                self.layers.append(AveragePooling2DLayer(pool_size))
            
            elif isinstance(layer, tf.keras.layers.Flatten):
                self.layers.append(FlattenLayer())
            
            elif isinstance(layer, tf.keras.layers.Dense):
                weights, bias = layer.get_weights()
                activation = layer.activation.__name__ if layer.activation else None
                self.layers.append(DenseLayer(weights, bias, activation))
            
            elif isinstance(layer, tf.keras.layers.Dropout):
                # Skip dropout during inference
                continue
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                # Skip batch normalization during inference for simplicity
                continue
        
        print(f"✓ Loaded {len(self.layers)} layers from Keras model")
    
    def forward(self, x):
        """Forward propagation through all layers"""
        output = x
        print(f"  Input shape: {output.shape}")
        
        for i, layer in enumerate(self.layers):
            try:
                prev_shape = output.shape
                output = layer.forward(output)
                print(f"  Layer {i+1} ({layer.__class__.__name__}): {prev_shape} → {output.shape}")
                
                # Check for problematic outputs
                if output.size == 0:
                    raise ValueError(f"Layer {i+1} produced empty output")
                    
            except Exception as e:
                print(f"  ❌ Error in layer {i+1} ({layer.__class__.__name__}): {e}")
                print(f"    Input shape: {prev_shape}")
                if hasattr(layer, 'weights'):
                    print(f"    Layer weights shape: {layer.weights.shape}")
                if hasattr(layer, 'bias'):
                    print(f"    Layer bias shape: {layer.bias.shape}")
                raise e
        
        return output
    
    def predict(self, x):
        """Make predictions using the scratch implementation"""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def save_weights(self, filepath):
        """Save scratch model weights"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.layers, f)
    
    def load_weights(self, filepath):
        """Load scratch model weights"""
        with open(filepath, 'rb') as f:
            self.layers = pickle.load(f)

def test_forward_propagation_cnn(saved_models, x_test, y_test):
    """Test CNN forward propagation from scratch vs Keras"""
    
    print("\n" + "="*60)
    print("TESTING CNN FORWARD PROPAGATION FROM SCRATCH")
    print("="*60)
    
    # Use a smaller test set for comparison
    x_test_small = x_test[:100]
    y_test_small = y_test[:100]
    
    comparison_results = []
    
    for experiment, model_info in saved_models.items():
        model_path = model_info['model']
        
        if os.path.exists(model_path):
            print(f"\n🧪 Testing {experiment} model...")
            
            try:
                # Load Keras model
                keras_model = keras.models.load_model(model_path)
                
                # Debug: Print model summary to understand architecture
                print(f"📋 Model architecture:")
                for i, layer in enumerate(keras_model.layers):
                    print(f"  Layer {i+1}: {layer.__class__.__name__} - {layer.output_shape}")
                
                # Get Keras predictions
                keras_pred = keras_model.predict(x_test_small, verbose=0)
                keras_classes = np.argmax(keras_pred, axis=1)
                
                # Test scratch implementation with detailed debugging
                print(f"\n🛠️  Creating scratch implementation...")
                cnn_scratch = CNNFromScratch()
                cnn_scratch.load_keras_model(keras_model)
                
                print(f"\n🧮 Running forward pass...")
                scratch_pred = cnn_scratch.forward(x_test_small)
                scratch_classes = np.argmax(scratch_pred, axis=1)
                
                # Compare results
                matches = np.sum(keras_classes == scratch_classes)
                match_percentage = matches / len(keras_classes) * 100
                
                # Calculate F1 scores
                keras_f1 = f1_score(y_test_small, keras_classes, average='macro')
                scratch_f1 = f1_score(y_test_small, scratch_classes, average='macro')
                
                # Calculate numerical differences
                prob_diff = np.mean(np.abs(keras_pred - scratch_pred))
                max_prob_diff = np.max(np.abs(keras_pred - scratch_pred))
                
                print(f"\n📊 Results for {experiment}:")
                print(f"  Keras vs Scratch predictions match: {matches}/{len(keras_classes)} ({match_percentage:.1f}%)")
                print(f"  Keras F1-Score: {keras_f1:.4f}")
                print(f"  Scratch F1-Score: {scratch_f1:.4f}")
                print(f"  F1-Score difference: {abs(keras_f1 - scratch_f1):.4f}")
                print(f"  Mean probability difference: {prob_diff:.6f}")
                print(f"  Max probability difference: {max_prob_diff:.6f}")
                
                # Show sample predictions
                print(f"\n🔍 Sample predictions (first 10):")
                print(f"  True:    {y_test_small[:10]}")
                print(f"  Keras:   {keras_classes[:10]}")
                print(f"  Scratch: {scratch_classes[:10]}")
                print(f"  Match:   {keras_classes[:10] == scratch_classes[:10]}")
                
                comparison_results.append({
                    'experiment': experiment,
                    'match_percentage': match_percentage,
                    'keras_f1': keras_f1,
                    'scratch_f1': scratch_f1,
                    'f1_diff': abs(keras_f1 - scratch_f1),
                    'prob_diff': prob_diff
                })
                
            except Exception as e:
                print(f"  ❌ Error testing {experiment}: {e}")
                print(f"  Input shape: {x_test_small.shape}")
                
                # Try to load model and inspect architecture for debugging
                try:
                    keras_model = keras.models.load_model(model_path)
                    print(f"  Model loaded successfully, layers:")
                    for i, layer in enumerate(keras_model.layers):
                        print(f"    {i}: {layer.__class__.__name__} - Input: {layer.input_shape}, Output: {layer.output_shape}")
                except Exception as load_error:
                    print(f"  Failed to load model for debugging: {load_error}")
                    
        else:
            print(f"  ❌ Model file not found: {model_path}")
    
    # Summary
    if comparison_results:
        print(f"\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        df_comparison = pd.DataFrame(comparison_results)
        avg_match = df_comparison['match_percentage'].mean()
        avg_f1_diff = df_comparison['f1_diff'].mean()
        
        print(f"Average prediction match: {avg_match:.1f}%")
        print(f"Average F1-score difference: {avg_f1_diff:.4f}")
        
        if avg_match >= 95.0:
            print("✅ EXCELLENT: Scratch implementation is highly accurate!")
        elif avg_match >= 90.0:
            print("🟢 GOOD: Minor differences, likely due to numerical precision")
        elif avg_match >= 80.0:
            print("🟡 ACCEPTABLE: Some differences, check implementation")
        else:
            print("🔴 NEEDS WORK: Significant differences found")
    
    return comparison_results
    """
    Compare Keras model predictions with scratch implementation
    
    Parameters:
    - keras_model: trained Keras model
    - x_test_sample: test data sample for comparison
    - y_test_sample: test labels for F1 score calculation
    - experiment_name: name of the experiment for reporting
    
    Returns:
    - comparison_results: dictionary with comparison metrics
    """
    print(f"\n{'='*60}")
    print(f"COMPARING KERAS VS SCRATCH IMPLEMENTATION - {experiment_name.upper()}")
    print(f"{'='*60}")
    
    # Get Keras predictions
    print("🔥 Getting Keras predictions...")
    keras_pred_probs = keras_model.predict(x_test_sample, verbose=0)
    keras_pred_classes = np.argmax(keras_pred_probs, axis=1)
    
    # Create and load scratch implementation
    print("🛠️  Creating scratch implementation...")
    cnn_scratch = CNNFromScratch()
    cnn_scratch.load_keras_model(keras_model)
    
    # Get scratch predictions
    print("🧮 Getting scratch predictions...")
    scratch_pred_probs = cnn_scratch.forward(x_test_sample)
    scratch_pred_classes = np.argmax(scratch_pred_probs, axis=1)
    
    # Compare predictions
    matches = np.sum(keras_pred_classes == scratch_pred_classes)
    total_samples = len(keras_pred_classes)
    match_percentage = (matches / total_samples) * 100
    
    # Calculate F1 scores
    keras_f1 = f1_score(y_test_sample, keras_pred_classes, average='macro')
    scratch_f1 = f1_score(y_test_sample, scratch_pred_classes, average='macro')
    
    # Calculate accuracy
    keras_accuracy = accuracy_score(y_test_sample, keras_pred_classes)
    scratch_accuracy = accuracy_score(y_test_sample, scratch_pred_classes)
    
    # Calculate numerical differences in outputs
    prob_diff = np.mean(np.abs(keras_pred_probs - scratch_pred_probs))
    max_prob_diff = np.max(np.abs(keras_pred_probs - scratch_pred_probs))
    
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"  Total test samples: {total_samples}")
    print(f"  Matching predictions: {matches}/{total_samples} ({match_percentage:.2f}%)")
    print(f"  ")
    print(f"  Keras Model:")
    print(f"    Accuracy: {keras_accuracy:.4f}")
    print(f"    Macro F1-Score: {keras_f1:.4f}")
    print(f"  ")
    print(f"  Scratch Implementation:")
    print(f"    Accuracy: {scratch_accuracy:.4f}")
    print(f"    Macro F1-Score: {scratch_f1:.4f}")
    print(f"  ")
    print(f"  Numerical Differences:")
    print(f"    Mean absolute difference in probabilities: {prob_diff:.6f}")
    print(f"    Max absolute difference in probabilities: {max_prob_diff:.6f}")
    
    # Show some example predictions
    print(f"\n🔍 Sample Predictions (first 10):")
    print(f"  True labels:    {y_test_sample[:10]}")
    print(f"  Keras preds:    {keras_pred_classes[:10]}")
    print(f"  Scratch preds:  {scratch_pred_classes[:10]}")
    print(f"  Match:          {keras_pred_classes[:10] == scratch_pred_classes[:10]}")
    
    # Assessment
    if match_percentage >= 99.0:
        assessment = "✅ EXCELLENT - Implementation matches very well!"
    elif match_percentage >= 95.0:
        assessment = "🟢 GOOD - Minor differences, likely due to numerical precision"
    elif match_percentage >= 90.0:
        assessment = "🟡 ACCEPTABLE - Some differences, check implementation"
    else:
        assessment = "🔴 POOR - Significant differences, implementation needs review"
    
    print(f"\n{assessment}")
    
    comparison_results = {
        'experiment': experiment_name,
        'total_samples': total_samples,
        'matching_predictions': matches,
        'match_percentage': match_percentage,
        'keras_accuracy': keras_accuracy,
        'keras_f1': keras_f1,
        'scratch_accuracy': scratch_accuracy,
        'scratch_f1': scratch_f1,
        'mean_prob_diff': prob_diff,
        'max_prob_diff': max_prob_diff,
        'assessment': assessment
    }
    
    return comparison_results

def plot_training_history(training_histories, experiment_name, save_dir='plots'):
    """
    Modular function to plot training and validation loss for different test cases
    
    Parameters:
    - training_histories: dict containing training histories for different variants
    - experiment_name: string name of the experiment
    - save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'CNN Experiment: {experiment_name} - Training History', fontsize=14)
    
    for key, history in training_histories.items():
        if experiment_name.lower().replace(' ', '_') in key:
            variant_name = key.replace(f'{experiment_name.lower().replace(" ", "_")}_', '').replace('_', ' ').title()
            epochs = range(1, len(history['loss']) + 1)
            
            ax1.plot(epochs, history['loss'], label=variant_name, marker='o', markersize=3)
            ax2.plot(epochs, history['val_loss'], label=variant_name, marker='o', markersize=3)
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f'{save_dir}/cnn_{experiment_name.lower().replace(" ", "_")}_training_history.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {plot_filename}")

def analyze_cnn_hyperparameters(return_models=False, compare_with_scratch=True):
    """CNN Hyperparameter Analysis with CIFAR-10"""
    
    # GPU optimization
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU acceleration for CNN training!")
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Load and prepare CIFAR-10 data
    (x_train_full, y_train_full), (x_test_original, y_test_original) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test_original = x_test_original.astype('float32') / 255.0
    
    print("Original Dataset Info:")
    print(f"Train Dataset Length: {len(x_train_full)}")
    print(f"Test Dataset Length: {len(x_test_original)}")
    
    # FIXED DATA SPLIT: 40k train, 10k validation (4:1 ratio), 10k test randomly chosen
    np.random.seed(42)
    
    # Split train_full into train and validation (4:1 ratio = 40k:10k)
    total_train_val = len(x_train_full)
    train_size = 40000
    val_size = 10000
    
    indices = np.random.permutation(total_train_val)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    
    # Randomly choose 10k from test set
    test_indices = np.random.choice(len(x_test_original), 10000, replace=False)
    
    # Create the splits
    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices].flatten()
    x_val = x_train_full[val_indices] 
    y_val = y_train_full[val_indices].flatten()
    x_test = x_test_original[test_indices]
    y_test = y_test_original[test_indices].flatten()
    
    print(f"\nCORRECTED Data Split (4:1 train:val ratio):")
    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Validation set: {x_val.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    print(f"Train:Val ratio = {x_train.shape[0]}:{x_val.shape[0]} = {x_train.shape[0]/x_val.shape[0]:.1f}:1")
    
    results = []
    training_histories = {}
    saved_models = {}
    comparison_results = []  # Store comparison results
    
    # Prepare test sample for comparison (use smaller batch for efficiency)
    x_test_sample = x_test[:100]  # Use first 100 test samples
    y_test_sample = y_test[:100]
    
    # ========== EXPERIMENT 1: Number of Convolutional Layers ==========
    print("\n" + "="*60)
    print("EXPERIMENT 1: TESTING NUMBER OF CONVOLUTIONAL LAYERS")
    print("="*60)
    
    layer_variants = [
        {
            'name': '2_conv_layers',
            'description': '2 Convolutional Layers',
            'layers': [
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(10, activation='softmax')
            ]
        },
        {
            'name': '4_conv_layers', 
            'description': '4 Convolutional Layers',
            'layers': [
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(10, activation='softmax')
            ]
        },
        {
            'name': '6_conv_layers',
            'description': '6 Convolutional Layers', 
            'layers': [
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(10, activation='softmax')
            ]
        }
    ]
    
    best_conv_model = None
    best_conv_f1 = 0
    
    for variant in layer_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        model = keras.Sequential(variant['layers'])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',  # FIXED: was categorical_crossentropy
                     metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                          epochs=15,
                          batch_size=128,
                          validation_data=(x_val, y_val),
                          verbose=1)
        
        training_histories[f"conv_layers_{variant['name']}"] = history.history
        
        # Evaluate on test set
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        if f1 > best_conv_f1:
            best_conv_f1 = f1
            best_conv_model = model
            best_conv_name = variant['name']
        
        results.append({
            'experiment': 'conv_layers',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Save best model
    if best_conv_model is not None:
        os.makedirs('models', exist_ok=True)
        os.makedirs('weights', exist_ok=True)
        
        model_filename = f'models/best_conv_layers_{best_conv_name}.h5'
        weights_filename = f'weights/best_conv_layers_{best_conv_name}_weights.h5'
        
        best_conv_model.save(model_filename)
        best_conv_model.save_weights(weights_filename)
        
        saved_models['conv_layers'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': best_conv_f1
        }
        print(f"\n✓ Best conv layers model saved: {model_filename}")
        
        # Compare with scratch implementation
        if compare_with_scratch:
            comparison_result = compare_keras_vs_scratch(
                best_conv_model, x_test_sample, y_test_sample, 
                f"conv_layers_{best_conv_name}"
            )
            comparison_results.append(comparison_result)
    
    # ========== EXPERIMENT 2: Number of Filters per Layer ==========
    print("\n" + "="*60)
    print("EXPERIMENT 2: TESTING NUMBER OF FILTERS PER LAYER")
    print("="*60)
    
    filter_variants = [
        {'name': 'small_filters_16_32_64', 'description': 'Small Filters (16-32-64)', 'filters': [16, 32, 64]},
        {'name': 'medium_filters_32_64_128', 'description': 'Medium Filters (32-64-128)', 'filters': [32, 64, 128]},
        {'name': 'large_filters_64_128_256', 'description': 'Large Filters (64-128-256)', 'filters': [64, 128, 256]}
    ]
    
    best_filter_model = None
    best_filter_f1 = 0
    
    for variant in filter_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        f1_filt, f2_filt, f3_filt = variant['filters']
        
        model = keras.Sequential([
            layers.Conv2D(f1_filt, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            layers.Conv2D(f1_filt, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(f2_filt, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(f2_filt, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(f3_filt, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(f3_filt, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',  # FIXED
                     metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                          epochs=15,
                          batch_size=128,
                          validation_data=(x_val, y_val),
                          verbose=1)
        
        training_histories[f"filter_count_{variant['name']}"] = history.history
        
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        if f1 > best_filter_f1:
            best_filter_f1 = f1
            best_filter_model = model
            best_filter_name = variant['name']
        
        results.append({
            'experiment': 'filter_count',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Save best filter model
    if best_filter_model is not None:
        model_filename = f'models/best_filter_count_{best_filter_name}.h5'
        weights_filename = f'weights/best_filter_count_{best_filter_name}_weights.h5'
        
        best_filter_model.save(model_filename)
        best_filter_model.save_weights(weights_filename)
        
        saved_models['filter_count'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': best_filter_f1
        }
        print(f"\n✓ Best filter count model saved: {model_filename}")
        
        # Compare with scratch implementation
        if compare_with_scratch:
            comparison_result = compare_keras_vs_scratch(
                best_filter_model, x_test_sample, y_test_sample, 
                f"filter_count_{best_filter_name}"
            )
            comparison_results.append(comparison_result)
    
    # ========== EXPERIMENT 3: Filter Size ==========
    print("\n" + "="*60)
    print("EXPERIMENT 3: TESTING FILTER SIZES")
    print("="*60)
    
    kernel_variants = [
        {'name': '3x3_kernels', 'description': '3x3 Kernels', 'kernel_size': (3, 3)},
        {'name': '5x5_kernels', 'description': '5x5 Kernels', 'kernel_size': (5, 5)},
        {'name': '7x7_kernels', 'description': '7x7 Kernels', 'kernel_size': (7, 7)}
    ]
    
    best_kernel_model = None
    best_kernel_f1 = 0
    
    for variant in kernel_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        kernel_size = variant['kernel_size']
        
        model = keras.Sequential([
            layers.Conv2D(32, kernel_size, activation='relu', input_shape=(32, 32, 3), padding='same'),
            layers.Conv2D(32, kernel_size, activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, kernel_size, activation='relu', padding='same'),
            layers.Conv2D(64, kernel_size, activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, kernel_size, activation='relu', padding='same'),
            layers.Conv2D(128, kernel_size, activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',  # FIXED
                     metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                          epochs=15,
                          batch_size=128,
                          validation_data=(x_val, y_val),
                          verbose=1)
        
        training_histories[f"kernel_size_{variant['name']}"] = history.history
        
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        if f1 > best_kernel_f1:
            best_kernel_f1 = f1
            best_kernel_model = model
            best_kernel_name = variant['name']
        
        results.append({
            'experiment': 'kernel_size',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Save best kernel model
    if best_kernel_model is not None:
        model_filename = f'models/best_kernel_size_{best_kernel_name}.h5'
        weights_filename = f'weights/best_kernel_size_{best_kernel_name}_weights.h5'
        
        best_kernel_model.save(model_filename)
        best_kernel_model.save_weights(weights_filename)
        
        saved_models['kernel_size'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': best_kernel_f1
        }
        print(f"\n✓ Best kernel size model saved: {model_filename}")
        
        # Compare with scratch implementation
        if compare_with_scratch:
            comparison_result = compare_keras_vs_scratch(
                best_kernel_model, x_test_sample, y_test_sample, 
                f"kernel_size_{best_kernel_name}"
            )
            comparison_results.append(comparison_result)
    
    # ========== EXPERIMENT 4: Pooling Layer Types ==========
    print("\n" + "="*60)
    print("EXPERIMENT 4: TESTING POOLING LAYER TYPES")
    print("="*60)
    
    pooling_variants = [
        {'name': 'max_pooling', 'description': 'Max Pooling'},
        {'name': 'average_pooling', 'description': 'Average Pooling'}
    ]
    
    best_pooling_model = None
    best_pooling_f1 = 0
    
    for variant in pooling_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        
        if variant['name'] == 'max_pooling':
            pooling_layer1 = layers.MaxPooling2D((2, 2))
            pooling_layer2 = layers.MaxPooling2D((2, 2))
            pooling_layer3 = layers.MaxPooling2D((2, 2))
        else:
            pooling_layer1 = layers.AveragePooling2D((2, 2))
            pooling_layer2 = layers.AveragePooling2D((2, 2))
            pooling_layer3 = layers.AveragePooling2D((2, 2))
        
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            pooling_layer1,
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            pooling_layer2,
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            pooling_layer3,
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',  # FIXED
                     metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                          epochs=15,
                          batch_size=128,
                          validation_data=(x_val, y_val),
                          verbose=1)
        
        training_histories[f"pooling_type_{variant['name']}"] = history.history
        
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        if f1 > best_pooling_f1:
            best_pooling_f1 = f1
            best_pooling_model = model
            best_pooling_name = variant['name']
        
        results.append({
            'experiment': 'pooling_type',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Save best pooling model
    if best_pooling_model is not None:
        model_filename = f'models/best_pooling_type_{best_pooling_name}.h5'
        weights_filename = f'weights/best_pooling_type_{best_pooling_name}_weights.h5'
        
        best_pooling_model.save(model_filename)
        best_pooling_model.save_weights(weights_filename)
        
        saved_models['pooling_type'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': best_pooling_f1
        }
        print(f"\n✓ Best pooling type model saved: {model_filename}")
        
        # Compare with scratch implementation
        if compare_with_scratch:
            comparison_result = compare_keras_vs_scratch(
                best_pooling_model, x_test_sample, y_test_sample, 
                f"pooling_type_{best_pooling_name}"
            )
            comparison_results.append(comparison_result)
    
    # Print results summary
    print("\n" + "="*60)
    print("CNN EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    for experiment in df_results['experiment'].unique():
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        
        print(f"\n{experiment.upper()} - Best Variant: {best_variant['variant']}")
        print(f"  Description: {best_variant['description']}")
        print(f"  F1-Score: {best_variant['f1_score']:.4f}")
        print(f"  Accuracy: {best_variant['accuracy']:.4f}")
        
        # Print conclusions based on experiment
        if experiment == 'conv_layers':
            print(f"  📊 CONCLUSION - Number of Convolutional Layers Impact:")
            print(f"     More convolutional layers generally improve feature extraction capability")
            print(f"     but may lead to overfitting and increased computational cost.")
        elif experiment == 'filter_count':
            print(f"  📊 CONCLUSION - Number of Filters Impact:")
            print(f"     More filters allow the network to learn more features")
            print(f"     but increase model complexity and training time.")
        elif experiment == 'kernel_size':
            print(f"  📊 CONCLUSION - Filter Size Impact:")
            print(f"     Larger filters capture more spatial context but may lose fine details.")
            print(f"     Smaller filters are better for detailed features but may miss larger patterns.")
        elif experiment == 'pooling_type':
            print(f"  📊 CONCLUSION - Pooling Type Impact:")
            print(f"     Max pooling preserves important features and is generally better for classification.")
            print(f"     Average pooling smooths features and may be better for certain texture tasks.")
    
    print(f"\nCNN Hyperparameter Analysis Complete!")
    print(f"✓ All model weights saved in weights/ directory")
    print(f"✓ Use plot_training_history() function to visualize training histories")
    
    # Print comparison summary if performed
    if compare_with_scratch and comparison_results:
        print(f"\n{'='*60}")
        print("KERAS VS SCRATCH IMPLEMENTATION SUMMARY")
        print(f"{'='*60}")
        
        df_comparison = pd.DataFrame(comparison_results)
        print(f"\nOverall Comparison Results:")
        print(f"Average match percentage: {df_comparison['match_percentage'].mean():.2f}%")
        print(f"Average F1 difference: {abs(df_comparison['keras_f1'] - df_comparison['scratch_f1']).mean():.4f}")
        print(f"Average accuracy difference: {abs(df_comparison['keras_accuracy'] - df_comparison['scratch_accuracy']).mean():.4f}")
        
        print(f"\nDetailed Results:")
        for _, row in df_comparison.iterrows():
            print(f"  {row['experiment']}: {row['match_percentage']:.1f}% match, F1 diff: {abs(row['keras_f1'] - row['scratch_f1']):.4f}")
    
    if return_models:
        return df_results, saved_models, training_histories, comparison_results if compare_with_scratch else None
    else:
        return df_results, training_histories, comparison_results if compare_with_scratch else None

if __name__ == "__main__":
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("\n🔥 Running CNN analysis with CIFAR-10...")
    results = analyze_cnn_hyperparameters(compare_with_scratch=True)
    
    if len(results) == 3:  # With comparison
        cnn_results, training_histories, comparison_results = results
    else:  # Without comparison
        cnn_results, training_histories = results
    
    # Example of how to use the plotting function
    experiments = ['conv_layers', 'filter_count', 'kernel_size', 'pooling_type']
    experiment_titles = [
        'Number of Convolutional Layers',
        'Number of Filters per Layer', 
        'Filter Size',
        'Pooling Layer Types'
    ]
    
    for exp, title in zip(experiments, experiment_titles):
        plot_training_history(training_histories, exp, 'plots')
    
    print("\nCNN Analysis Complete!")