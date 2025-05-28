import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

class EmbeddingLayer:
    def __init__(self, weights):
        self.weights = weights
    
    def forward(self, x):
        batch_size, seq_length = x.shape
        embedding_dim = self.weights.shape[1]
        
        output = np.zeros((batch_size, seq_length, embedding_dim))
        
        for b in range(batch_size):
            for s in range(seq_length):
                token_id = int(x[b, s])
                if token_id < self.weights.shape[0]:
                    output[b, s, :] = self.weights[token_id]
        
        return output

class SimpleRNNCell:
    def __init__(self, input_weights, recurrent_weights, bias):
        self.Wxh = input_weights     
        self.Whh = recurrent_weights 
        self.bias = bias             
    
    def forward(self, x, h_prev):
        h_new = np.tanh(np.dot(x, self.Wxh) + np.dot(h_prev, self.Whh) + self.bias)
        return h_new

class SimpleRNNLayer:
    def __init__(self, weights, bias, return_sequences=False):
        input_weights = weights[0]     
        recurrent_weights = weights[1]  
        
        self.cell = SimpleRNNCell(input_weights, recurrent_weights, bias)
        self.return_sequences = return_sequences
        self.hidden_size = recurrent_weights.shape[0]
    
    def forward(self, x):
        batch_size, seq_length, input_size = x.shape
        
        h = np.zeros((batch_size, self.hidden_size))
        
        if self.return_sequences:
            outputs = np.zeros((batch_size, seq_length, self.hidden_size))
            for t in range(seq_length):
                h = self.cell.forward(x[:, t, :], h)
                outputs[:, t, :] = h
            return outputs
        else:
            for t in range(seq_length):
                h = self.cell.forward(x[:, t, :], h)
            return h

class BidirectionalRNNLayer:
    def __init__(self, forward_weights, forward_bias, backward_weights, backward_bias, return_sequences=False):
        self.forward_rnn = SimpleRNNLayer(forward_weights, forward_bias, True)
        self.backward_rnn = SimpleRNNLayer(backward_weights, backward_bias, True)
        self.return_sequences = return_sequences
    
    def forward(self, x):
        forward_output = self.forward_rnn.forward(x)
        
        x_reversed = x[:, ::-1, :]
        backward_output = self.backward_rnn.forward(x_reversed)
        backward_output = backward_output[:, ::-1, :]
        
        output = np.concatenate([forward_output, backward_output], axis=-1)
        
        if not self.return_sequences:
            return output[:, -1, :]
        
        return output

class DropoutLayer:
    def __init__(self, rate=0.5):
        self.rate = rate
    
    def forward(self, x, training=False):
        if not training:
            return x
        
        mask = np.random.binomial(1, 1 - self.rate, x.shape) / (1 - self.rate)
        return x * mask

class DenseLayer:
    def __init__(self, weights, bias, activation=None):
        self.weights = weights
        self.bias = bias     
        self.activation = activation
    
    def forward(self, x):
        output = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'softmax':
            exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
            output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return output

class RNNFromScratch:
    def __init__(self):
        self.layers = []
    
    def load_keras_model(self, model_path):
        keras_model = keras.models.load_model(model_path)
        self.layers = []
        
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Embedding):
                weights = layer.get_weights()[0]
                self.layers.append(EmbeddingLayer(weights))
            
            elif isinstance(layer, tf.keras.layers.SimpleRNN):
                weights = layer.get_weights()
                bias = weights[2] if len(weights) > 2 else np.zeros(weights[1].shape[0])
                return_sequences = layer.return_sequences
                self.layers.append(SimpleRNNLayer([weights[0], weights[1]], bias, return_sequences))
            
            elif isinstance(layer, tf.keras.layers.Bidirectional):
                rnn_layer = layer.layer
                if isinstance(rnn_layer, tf.keras.layers.SimpleRNN):
                    weights = layer.get_weights()
                    num_weights_per_direction = len(weights) // 2
                    forward_weights = weights[:num_weights_per_direction]
                    backward_weights = weights[num_weights_per_direction:]
                    
                    forward_bias = forward_weights[2] if len(forward_weights) > 2 else np.zeros(forward_weights[1].shape[0])
                    backward_bias = backward_weights[2] if len(backward_weights) > 2 else np.zeros(backward_weights[1].shape[0])
                    
                    return_sequences = rnn_layer.return_sequences
                    self.layers.append(BidirectionalRNNLayer(
                        [forward_weights[0], forward_weights[1]], forward_bias,
                        [backward_weights[0], backward_weights[1]], backward_bias,
                        return_sequences
                    ))
            
            elif isinstance(layer, tf.keras.layers.Dropout):
                rate = layer.rate
                self.layers.append(DropoutLayer(rate))
            
            elif isinstance(layer, tf.keras.layers.Dense):
                weights, bias = layer.get_weights()
                activation = layer.activation.__name__ if layer.activation else None
                self.layers.append(DenseLayer(weights, bias, activation))
        
        print(f"Loaded {len(self.layers)} layers from Keras model")
    
    def forward(self, x, training=False):
        output = x
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DropoutLayer):
                output = layer.forward(output, training)
            else:
                output = layer.forward(output)
            print(f"Layer {i} ({type(layer).__name__}) output shape: {output.shape}")
        
        return output
    
    def predict(self, x):
        logits = self.forward(x, training=False)
        return np.argmax(logits, axis=1)
    
    def predict_proba(self, x):
        return self.forward(x, training=False)
    
    def save_weights(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.layers, f)
    
    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            self.layers = pickle.load(f)

def test_rnn_scratch():
    # dummy data
    batch_size = 10
    seq_length = 20
    vocab_size = 1000
    
    # Generate random token sequences
    x_test = np.random.randint(0, vocab_size, (batch_size, seq_length))
    y_test = np.random.randint(0, 3, batch_size)
    
    try:
        # laod model
        keras_model = keras.models.load_model('models/rnn_layers_2.h5')
        keras_pred = keras_model.predict(x_test)
        keras_pred_classes = np.argmax(keras_pred, axis=1)
        
        rnn_scratch = RNNFromScratch()
        rnn_scratch.load_keras_model('models/rnn_layers_2.h5')
        
        # forward propagation
        scratch_pred = rnn_scratch.forward(x_test)
        scratch_pred_classes = np.argmax(scratch_pred, axis=1)
        
        print("=== RNN COMPARISON RESULTS ===")
        print(f"Keras predictions: {keras_pred_classes}")
        print(f"Scratch predictions: {scratch_pred_classes}")
        
        # matching prediction
        matches = np.sum(keras_pred_classes == scratch_pred_classes)
        print(f"Matching predictions: {matches}/{len(keras_pred_classes)} ({matches/len(keras_pred_classes)*100:.2f}%)")
        
        return rnn_scratch
        
    except FileNotFoundError:
        print("Model file not found. Please run RNN training first.")
        return None

if __name__ == "__main__":
    test_rnn_scratch()
