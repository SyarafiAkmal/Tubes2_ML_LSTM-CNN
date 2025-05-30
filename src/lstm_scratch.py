import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

class LSTMCell:
    def __init__(self, input_weights, recurrent_weights, bias):
        self.Wi, self.Wf, self.Wc, self.Wo = np.split(input_weights, 4, axis=1)
        self.Ui, self.Uf, self.Uc, self.Uo = np.split(recurrent_weights, 4, axis=1)
        self.bi, self.bf, self.bc, self.bo = np.split(bias, 4)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, x, h_prev, c_prev):
        # Input gate
        i_t = self.sigmoid(np.dot(x, self.Wi) + np.dot(h_prev, self.Ui) + self.bi)
        
        # Forget gate
        f_t = self.sigmoid(np.dot(x, self.Wf) + np.dot(h_prev, self.Uf) + self.bf)
        
        # Candidate values
        c_tilde = self.tanh(np.dot(x, self.Wc) + np.dot(h_prev, self.Uc) + self.bc)
        
        # Output gate
        o_t = self.sigmoid(np.dot(x, self.Wo) + np.dot(h_prev, self.Uo) + self.bo)
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Update hidden state
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t

class LSTMLayer:
    def __init__(self, weights, bias, return_sequences=False):
        # Extract weights
        input_weights = weights[0]      # Shape: (input_size, 4 * hidden_size)
        recurrent_weights = weights[1]  # Shape: (hidden_size, 4 * hidden_size)
        
        self.cell = LSTMCell(input_weights, recurrent_weights, bias)
        self.return_sequences = return_sequences
        self.hidden_size = recurrent_weights.shape[0]
    
    def forward(self, x):
        batch_size, seq_length, input_size = x.shape
        
        # Initialize hidden and cell states
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        if self.return_sequences:
            # Return all hidden states
            outputs = np.zeros((batch_size, seq_length, self.hidden_size))
            for t in range(seq_length):
                h, c = self.cell.forward(x[:, t, :], h, c)
                outputs[:, t, :] = h
            return outputs
        else:
            # Return only last hidden state
            for t in range(seq_length):
                h, c = self.cell.forward(x[:, t, :], h, c)
            return h

class BidirectionalLSTMLayer:
    def __init__(self, forward_weights, forward_bias, backward_weights, backward_bias, return_sequences=False):
        self.forward_lstm = LSTMLayer(forward_weights, forward_bias, True)  # Always return sequences for bidirectional
        self.backward_lstm = LSTMLayer(backward_weights, backward_bias, True)
        self.return_sequences = return_sequences
    
    def forward(self, x):
        # Forward direction
        forward_output = self.forward_lstm.forward(x)
        
        # Backward direction (reverse input sequence)
        x_reversed = x[:, ::-1, :]
        backward_output = self.backward_lstm.forward(x_reversed)
        backward_output = backward_output[:, ::-1, :]  # Reverse back
        
        # Concatenate forward and backward outputs
        output = np.concatenate([forward_output, backward_output], axis=-1)
        
        if not self.return_sequences:
            # Return only last time step
            return output[:, -1, :]
        
        return output

class EmbeddingLayer:
    def __init__(self, weights):
        self.weights = weights  # Shape: (vocab_size, embedding_dim)
    
    def forward(self, x):
        batch_size, seq_length = x.shape
        embedding_dim = self.weights.shape[1]
        
        # Initialize output
        output = np.zeros((batch_size, seq_length, embedding_dim))
        
        # Lookup embeddings
        for b in range(batch_size):
            for s in range(seq_length):
                token_id = int(x[b, s])
                if token_id < self.weights.shape[0]:
                    output[b, s, :] = self.weights[token_id]
        
        return output

class DropoutLayer:
    def __init__(self, rate=0.5):
        self.rate = rate
    
    def forward(self, x, training=False):
        if not training:
            return x
        
        # During training, randomly set some neurons to zero
        mask = np.random.binomial(1, 1 - self.rate, x.shape) / (1 - self.rate)
        return x * mask

class DenseLayer:
    def __init__(self, weights, bias, activation=None):
        self.weights = weights  # Shape: (input_size, output_size)
        self.bias = bias        # Shape: (output_size,)
        self.activation = activation
    
    def forward(self, x):
        output = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'softmax':
            # Stable softmax
            exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
            output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return output

class LSTMFromScratch:
    def __init__(self):
        self.layers = []
    
    def load_keras_model(self, model_path):
        keras_model = keras.models.load_model(model_path)
        self.layers = []
        
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Embedding):
                weights = layer.get_weights()[0]
                self.layers.append(EmbeddingLayer(weights))
            
            elif isinstance(layer, tf.keras.layers.LSTM):
                weights = layer.get_weights()
                bias = weights[2] if len(weights) > 2 else np.zeros(weights[1].shape[0])
                return_sequences = layer.return_sequences
                self.layers.append(LSTMLayer([weights[0], weights[1]], bias, return_sequences))
            
            elif isinstance(layer, tf.keras.layers.Bidirectional):
                # Get the wrapped LSTM layer
                lstm_layer = layer.layer
                if isinstance(lstm_layer, tf.keras.layers.LSTM):
                    # Bidirectional LSTM has forward and backward weights
                    weights = layer.get_weights()
                    # Split weights for forward and backward
                    num_weights_per_direction = len(weights) // 2
                    forward_weights = weights[:num_weights_per_direction]
                    backward_weights = weights[num_weights_per_direction:]
                    
                    forward_bias = forward_weights[2] if len(forward_weights) > 2 else np.zeros(forward_weights[1].shape[0])
                    backward_bias = backward_weights[2] if len(backward_weights) > 2 else np.zeros(backward_weights[1].shape[0])
                    
                    return_sequences = lstm_layer.return_sequences
                    self.layers.append(BidirectionalLSTMLayer(
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
    
    def save_weights(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.layers, f)
    
    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            self.layers = pickle.load(f)

def test_lstm_scratch():
    from sklearn.metrics import f1_score, accuracy_score
    
    # Create synthetic test data
    batch_size = 10
    seq_length = 20
    vocab_size = 1000
    
    # Generate random token sequences
    x_test = np.random.randint(0, vocab_size, (batch_size, seq_length))
    y_test = np.random.randint(0, 3, batch_size)  # 3 classes
    
    try:
        # Load Keras model
        keras_model = keras.models.load_model('models/lstm_layers_2.h5')
        keras_pred = keras_model.predict(x_test)
        keras_pred_classes = np.argmax(keras_pred, axis=1)
        
        # Test scratch implementation
        lstm_scratch = LSTMFromScratch()
        lstm_scratch.load_keras_model('models/lstm_layers_2.h5')
        
        # Forward propagation
        scratch_pred = lstm_scratch.forward(x_test)
        scratch_pred_classes = np.argmax(scratch_pred, axis=1)
        
        # Compare results
        print("=== LSTM COMPARISON RESULTS ===")
        print(f"Keras predictions: {keras_pred_classes}")
        print(f"Scratch predictions: {scratch_pred_classes}")
        
        # Check if predictions match
        matches = np.sum(keras_pred_classes == scratch_pred_classes)
        print(f"Matching predictions: {matches}/{len(keras_pred_classes)} ({matches/len(keras_pred_classes)*100:.2f}%)")
        
        return lstm_scratch
        
    except FileNotFoundError:
        print("Model file not found. Please run LSTM training first.")
        return None

if __name__ == "__main__":
    test_lstm_scratch()
