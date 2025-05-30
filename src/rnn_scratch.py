import numpy as np
import tensorflow as tf
from tensorflow import keras

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
        
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'softmax':
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
                weights = layer.get_weights()

                forward_weights = [weights[0], weights[1]]
                forward_bias = weights[2]
                backward_weights = [weights[3], weights[4]]
                backward_bias = weights[5]

                forward_bias = forward_weights[2] if len(forward_weights) > 2 else np.zeros(forward_weights[1].shape[0])
                backward_bias = backward_weights[2] if len(backward_weights) > 2 else np.zeros(backward_weights[1].shape[0])

                return_sequences = layer.return_sequences

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
        
        print(f"✓ Loaded {len(self.layers)} layers from Keras model")
    
    def forward(self, x, training=False):
        output = x
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DropoutLayer):
                output = layer.forward(output, training)
            else:
                output = layer.forward(output)
        
        return output
    
    def predict(self, x):
        logits = self.forward(x, training=False)
        return np.argmax(logits, axis=1)
    
    def predict_proba(self, x):
        return self.forward(x, training=False)
