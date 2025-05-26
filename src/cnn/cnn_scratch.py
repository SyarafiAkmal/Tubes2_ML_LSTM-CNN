import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

class Conv2DLayer:
    def __init__(self, weights, bias, activation='relu'):
        self.weights = weights  # Shape: (kernel_h, kernel_w, input_channels, output_channels)
        self.bias = bias        # Shape: (output_channels,)
        self.activation = activation
    
    def forward(self, x):
        batch_size, input_h, input_w, input_c = x.shape
        kernel_h, kernel_w, _, output_c = self.weights.shape
        
        # Calculate output dimensions
        output_h = input_h - kernel_h + 1
        output_w = input_w - kernel_w + 1
        
        # Initialize output
        output = np.zeros((batch_size, output_h, output_w, output_c))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(output_c):
                for oh in range(output_h):
                    for ow in range(output_w):
                        # Extract patch
                        patch = x[b, oh:oh+kernel_h, ow:ow+kernel_w, :]
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
        
        output_h = input_h // pool_h
        output_w = input_w // pool_w
        
        output = np.zeros((batch_size, output_h, output_w, channels))
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(output_h):
                    for ow in range(output_w):
                        h_start = oh * pool_h
                        h_end = h_start + pool_h
                        w_start = ow * pool_w
                        w_end = w_start + pool_w
                        
                        pool_region = x[b, h_start:h_end, w_start:w_end, c]
                        output[b, oh, ow, c] = np.max(pool_region)
        
        return output

class AveragePooling2DLayer:
    def __init__(self, pool_size=(2, 2)):
        self.pool_size = pool_size
    
    def forward(self, x):
        batch_size, input_h, input_w, channels = x.shape
        pool_h, pool_w = self.pool_size
        
        output_h = input_h // pool_h
        output_w = input_w // pool_w
        
        output = np.zeros((batch_size, output_h, output_w, channels))
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(output_h):
                    for ow in range(output_w):
                        h_start = oh * pool_h
                        h_end = h_start + pool_h
                        w_start = ow * pool_w
                        w_end = w_start + pool_w
                        
                        pool_region = x[b, h_start:h_end, w_start:w_end, c]
                        output[b, oh, ow, c] = np.mean(pool_region)
        
        return output

class FlattenLayer:
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

class DenseLayer:
    def __init__(self, weights, bias, activation=None):
        self.weights = weights  # Shape: (input_size, output_size)
        self.bias = bias        # Shape: (output_size,)
        self.activation = activation
    
    def forward(self, x):
        output = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'softmax':
            # Stable softmax
            exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
            output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return output

class CNNFromScratch:
    def __init__(self):
        self.layers = []
    
    def load_keras_model(self, model_path):
        keras_model = keras.models.load_model(model_path)
        self.layers = []
        
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights, bias = layer.get_weights()
                activation = layer.activation.__name__ if layer.activation else None
                self.layers.append(Conv2DLayer(weights, bias, activation))
            
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
        
        print(f"Loaded {len(self.layers)} layers from Keras model")
    
    def forward(self, x):
        output = x
        
        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
            print(f"Layer {i} output shape: {output.shape}")
        
        return output
    
    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def save_weights(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.layers, f)
    
    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            self.layers = pickle.load(f)

def test_cnn_scratch():
    from sklearn.metrics import f1_score, accuracy_score
    
    # Load test data
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.flatten()
    
    # Use small subset for testing
    x_test_small = x_test[:100]
    y_test_small = y_test[:100]
    
    # Load Keras model
    keras_model = keras.models.load_model('models/cnn_layers_3.h5')
    keras_pred = keras_model.predict(x_test_small)
    keras_pred_classes = np.argmax(keras_pred, axis=1)
    
    # Test scratch implementation
    cnn_scratch = CNNFromScratch()
    cnn_scratch.load_keras_model('models/cnn_layers_3.h5')
    
    # Forward propagation
    scratch_pred = cnn_scratch.forward(x_test_small)
    scratch_pred_classes = np.argmax(scratch_pred, axis=1)
    
    # Compare results
    print("=== COMPARISON RESULTS ===")
    print(f"Keras predictions: {keras_pred_classes[:10]}")
    print(f"Scratch predictions: {scratch_pred_classes[:10]}")
    
    # Calculate metrics
    keras_f1 = f1_score(y_test_small, keras_pred_classes, average='macro')
    scratch_f1 = f1_score(y_test_small, scratch_pred_classes, average='macro')
    
    print(f"\nKeras F1-Score: {keras_f1:.4f}")
    print(f"Scratch F1-Score: {scratch_f1:.4f}")
    
    # Check if predictions match
    matches = np.sum(keras_pred_classes == scratch_pred_classes)
    print(f"Matching predictions: {matches}/{len(keras_pred_classes)} ({matches/len(keras_pred_classes)*100:.2f}%)")
    
    return cnn_scratch

if __name__ == "__main__":
    test_cnn_scratch()
