import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

# Import scratch implementations
from rnn_scratch import RNNFromScratch
from utils.dataset_loader import DatasetLoader

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_rnn_hyperparameters(return_models=False):
    """RNN Hyperparameter Analysis with NusaX-Sentiment Dataset"""
    
    print("\n" + "="*60)
    print("RNN HYPERPARAMETER ANALYSIS - NUSAX-SENTIMENT")
    print("="*60)
    
    try:
        # Load NusaX-Sentiment dataset
        loader = DatasetLoader()
        df = loader.download_nusax_sentiment()
        X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_text_data(df)
        
        print(f"Dataset loaded successfully!")
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Classes: {np.unique(y_train)} (0=negative, 1=neutral, 2=positive)")
        
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure NusaX-Sentiment CSV files are in data/ directory")
        return {}
    
    # Text preprocessing
    max_features = 10000
    sequence_length = 100
    
    vectorizer = TextVectorization(
        max_tokens=max_features,
        output_sequence_length=sequence_length,
        output_mode='int'
    )
    
    vectorizer.adapt(X_train)
    
    X_train_vec = vectorizer(X_train)
    X_val_vec = vectorizer(X_val)
    X_test_vec = vectorizer(X_test)
    
    results = []
    training_histories = {}
    saved_models = {}
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    
    # ========== EXPERIMENT 1: Number of RNN Layers ==========
    print("\n" + "="*50)
    print("EXPERIMENT 1: TESTING NUMBER OF RNN LAYERS")
    print("="*50)
    
    layer_variants = [
        {'name': '1_rnn_layer', 'description': '1 RNN Layer', 'num_layers': 1},
        {'name': '2_rnn_layers', 'description': '2 RNN Layers', 'num_layers': 2},
        {'name': '3_rnn_layers', 'description': '3 RNN Layers', 'num_layers': 3}
    ]
    
    best_layer_f1 = 0
    best_layer_name = ""
    
    for variant in layer_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential()
        model.add(layers.Embedding(max_features, 128, input_length=sequence_length))
        
        # Add RNN layers
        for i in range(variant['num_layers']):
            return_sequences = (i < variant['num_layers'] - 1)  # Last layer returns sequence=False
            model.add(layers.SimpleRNN(64, return_sequences=return_sequences))
            if i < variant['num_layers'] - 1:  # Don't add dropout after last RNN layer
                model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(3, activation='softmax'))  # 3 classes
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        history = model.fit(X_train_vec, y_train,
                          epochs=10,
                          batch_size=32,
                          validation_data=(X_val_vec, y_val),
                          verbose=1)
        
        training_histories[f"rnn_layers_{variant['name']}"] = history.history
        
        # Evaluate on test set
        y_pred = model.predict(X_test_vec, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        # Track best for summary
        if f1 > best_layer_f1:
            best_layer_f1 = f1
            best_layer_name = variant['name']
        
        results.append({
            'experiment': 'rnn_layers',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        # SAVE EVERY MODEL (not just best)
        model_filename = f'models/rnn_layers_{variant["name"]}.h5'
        weights_filename = f'weights/rnn_layers_{variant["name"]}.weights.h5'
        
        model.save(model_filename)
        model.save_weights(weights_filename)
        
        saved_models[f'rnn_layers_{variant["name"]}'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': f1,
            'experiment': 'rnn_layers',
            'variant': variant['name']
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"  ✓ Model saved: {model_filename}")
    
    # ========== EXPERIMENT 2: Number of Cells per Layer ==========
    print("\n" + "="*50)
    print("EXPERIMENT 2: TESTING NUMBER OF CELLS PER LAYER")
    print("="*50)
    
    cell_variants = [
        {'name': '32_cells', 'description': '32 Cells per Layer', 'cells': 32},
        {'name': '64_cells', 'description': '64 Cells per Layer', 'cells': 64},
        {'name': '128_cells', 'description': '128 Cells per Layer', 'cells': 128}
    ]
    
    best_cell_f1 = 0
    best_cell_name = ""
    
    for variant in cell_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential([
            layers.Embedding(max_features, 128, input_length=sequence_length),
            layers.SimpleRNN(variant['cells'], return_sequences=True),
            layers.Dropout(0.3),
            layers.SimpleRNN(variant['cells']),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        history = model.fit(X_train_vec, y_train,
                          epochs=10,
                          batch_size=32,
                          validation_data=(X_val_vec, y_val),
                          verbose=1)
        
        training_histories[f"rnn_cells_{variant['name']}"] = history.history
        
        y_pred = model.predict(X_test_vec, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        # Track best for summary
        if f1 > best_cell_f1:
            best_cell_f1 = f1
            best_cell_name = variant['name']
        
        results.append({
            'experiment': 'rnn_cells',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        # SAVE EVERY MODEL (not just best)
        model_filename = f'models/rnn_cells_{variant["name"]}.h5'
        weights_filename = f'weights/rnn_cells_{variant["name"]}.weights.h5'
        
        model.save(model_filename)
        model.save_weights(weights_filename)
        
        saved_models[f'rnn_cells_{variant["name"]}'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': f1,
            'experiment': 'rnn_cells',
            'variant': variant['name']
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"  ✓ Model saved: {model_filename}")
    
    # ========== EXPERIMENT 3: Bidirectional vs Unidirectional ==========
    print("\n" + "="*50)
    print("EXPERIMENT 3: TESTING BIDIRECTIONAL vs UNIDIRECTIONAL")
    print("="*50)
    
    direction_variants = [
        {'name': 'unidirectional', 'description': 'Unidirectional RNN'},
        {'name': 'bidirectional', 'description': 'Bidirectional RNN'}
    ]
    
    best_direction_f1 = 0
    best_direction_name = ""
    
    for variant in direction_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential()
        model.add(layers.Embedding(max_features, 128, input_length=sequence_length))
        
        if variant['name'] == 'bidirectional':
            model.add(layers.Bidirectional(layers.SimpleRNN(64, return_sequences=True)))
            model.add(layers.Dropout(0.3))
            model.add(layers.Bidirectional(layers.SimpleRNN(64)))
        else:
            model.add(layers.SimpleRNN(64, return_sequences=True))
            model.add(layers.Dropout(0.3))
            model.add(layers.SimpleRNN(64))
        
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(3, activation='softmax'))
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        history = model.fit(X_train_vec, y_train,
                          epochs=10,
                          batch_size=32,
                          validation_data=(X_val_vec, y_val),
                          verbose=1)
        
        training_histories[f"rnn_direction_{variant['name']}"] = history.history
        
        y_pred = model.predict(X_test_vec, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        # Track best for summary
        if f1 > best_direction_f1:
            best_direction_f1 = f1
            best_direction_name = variant['name']
        
        results.append({
            'experiment': 'rnn_direction',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        # SAVE EVERY MODEL (not just best)
        model_filename = f'models/rnn_direction_{variant["name"]}.h5'
        weights_filename = f'weights/rnn_direction_{variant["name"]}.weights.h5'
        
        model.save(model_filename)
        model.save_weights(weights_filename)
        
        saved_models[f'rnn_direction_{variant["name"]}'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': f1,
            'experiment': 'rnn_direction',
            'variant': variant['name']
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"  ✓ Model saved: {model_filename}")
    
    # ========== FORWARD PROPAGATION FROM SCRATCH TESTING ==========
    print("\n" + "="*60)
    print("TESTING FORWARD PROPAGATION FROM SCRATCH")
    print("="*60)
    
    # Test RNN from scratch implementation
    test_forward_propagation_rnn(saved_models, X_test_vec, y_test)
    
    # ========== PLOTTING ==========
    print("\n" + "="*50)
    print("PLOTTING RNN TRAINING HISTORIES")
    print("="*50)
    
    experiments = ['rnn_layers', 'rnn_cells', 'rnn_direction']
    experiment_titles = [
        'Number of RNN Layers',
        'Number of Cells per Layer',
        'Bidirectional vs Unidirectional'
    ]
    
    for exp, title in zip(experiments, experiment_titles):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'RNN Experiment: {title} - Training History', fontsize=14)
        
        for key, history in training_histories.items():
            if exp in key:
                variant_name = key.replace(f'{exp}_', '').replace('_', ' ').title()
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
        plt.savefig(f'plots/rnn_{exp}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Print results summary
    print("\n" + "="*50)
    print("RNN EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    
    df_results = pd.DataFrame(results)
    
    for experiment in df_results['experiment'].unique():
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        
        print(f"\n{experiment.upper()} - Best Variant: {best_variant['variant']}")
        print(f"  Description: {best_variant['description']}")
        print(f"  F1-Score: {best_variant['f1_score']:.4f}")
        print(f"  Accuracy: {best_variant['accuracy']:.4f}")
        
        # Print conclusions based on experiment
        if experiment == 'rnn_layers':
            print(f"  📊 CONCLUSION - RNN Layers Impact:")
            print(f"     Adding more RNN layers increases the model's capacity to learn")
            print(f"     sequential patterns but may lead to vanishing gradient problems.")
        elif experiment == 'rnn_cells':
            print(f"  📊 CONCLUSION - RNN Cells Impact:")
            print(f"     More cells increase the model's representational capacity")
            print(f"     but require more data and computational resources.")
        elif experiment == 'rnn_direction':
            print(f"  📊 CONCLUSION - RNN Direction Impact:")
            print(f"     Bidirectional RNNs capture context from both past and future")
            print(f"     but double the parameters and cannot be used for real-time tasks.")
    
    print(f"\n" + "="*60)
    print("ALL SAVED MODELS SUMMARY")
    print("="*60)
    total_models = len(saved_models)
    print(f"Total models saved: {total_models}")
    
    for model_key, info in saved_models.items():
        print(f"✓ {model_key}: F1={info['f1_score']:.4f} | {info['model']}")
    
    print(f"\nRNN Hyperparameter Analysis Complete!")
    print(f"✓ All {total_models} model weights saved in weights/ directory")
    
    if return_models:
        return df_results, saved_models
    else:
        return df_results

def test_forward_propagation_rnn(saved_models, X_test_vec, y_test):
    """Test RNN forward propagation from scratch vs Keras"""
    
    print("\n--- Testing RNN Forward Propagation From Scratch ---")
    
    # Use a smaller test set for comparison
    X_test_small = X_test_vec[:50]
    y_test_small = y_test[:50]
    
    # Test a few representative models (not all to avoid too much output)
    test_models = ['rnn_layers_2_rnn_layers', 'rnn_cells_64_cells', 'rnn_direction_unidirectional']
    
    for model_key in test_models:
        if model_key in saved_models:
            model_info = saved_models[model_key]
            model_path = model_info['model']
            
            if os.path.exists(model_path):
                print(f"\nTesting {model_key} model...")
                
                try:
                    # Load Keras model
                    keras_model = keras.models.load_model(model_path)
                    keras_pred = keras_model.predict(X_test_small, verbose=0)
                    keras_classes = np.argmax(keras_pred, axis=1)
                    
                    # Test scratch implementation
                    rnn_scratch = RNNFromScratch()
                    rnn_scratch.load_keras_model(model_path)
                    
                    scratch_pred = rnn_scratch.forward(X_test_small)
                    scratch_classes = np.argmax(scratch_pred, axis=1)
                    
                    # Compare results
                    matches = np.sum(keras_classes == scratch_classes)
                    match_percentage = matches / len(keras_classes) * 100
                    
                    # Calculate F1 scores
                    keras_f1 = f1_score(y_test_small, keras_classes, average='macro')
                    scratch_f1 = f1_score(y_test_small, scratch_classes, average='macro')
                    
                    print(f"  Keras vs Scratch predictions match: {matches}/{len(keras_classes)} ({match_percentage:.1f}%)")
                    print(f"  Keras F1-Score: {keras_f1:.4f}")
                    print(f"  Scratch F1-Score: {scratch_f1:.4f}")
                    print(f"  F1-Score difference: {abs(keras_f1 - scratch_f1):.4f}")
                    
                except Exception as e:
                    print(f"  Error testing {model_key}: {e}")
            else:
                print(f"  Model file not found: {model_path}")

if __name__ == "__main__":
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("\n🔥 Running RNN analysis with NusaX-Sentiment...")
    rnn_results = analyze_rnn_hyperparameters()
    
    print("\nRNN Analysis Complete!")