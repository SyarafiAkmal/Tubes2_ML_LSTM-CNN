import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from rnn_scratch import RNNFromScratch
from utils.dataset_loader import DatasetLoader

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def create_rnn_model(config):
    model = keras.Sequential()
    
    # Embedding layer
    model.add(layers.Embedding(
        config['max_features'], 
        config['embedding_dim'], 
        input_length=config['sequence_length']
    ))
    
    # RNN layers
    for i in range(config['num_rnn_layers']):
        return_sequences = (i < config['num_rnn_layers'] - 1)
        
        if config['bidirectional']:
            model.add(layers.Bidirectional(
                layers.SimpleRNN(config['rnn_units'], return_sequences=return_sequences)
            ))
        else:
            model.add(layers.SimpleRNN(config['rnn_units'], return_sequences=return_sequences))
        
        if i < config['num_rnn_layers'] - 1:
            model.add(layers.Dropout(0.3))
    
    # Dense layers
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))
    
    return model

def analyze_rnn_hyperparameters():    
    print("\n" + "="*60)
    print("RNN HYPERPARAMETER ANALYSIS - NUSAX-SENTIMENT")
    print("="*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load dataset
    loader = DatasetLoader()
    df = loader.download_nusax_sentiment()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_text_data(df)
    
    print(f"  Dataset loaded successfully!")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Classes: {np.unique(y_train)} (0=negative, 1=neutral, 2=positive)")
    
    # Text preprocessing
    max_features = 10000
    sequence_length = 100
    embedding_dim = 128
    
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
    
    # Base configuration
    base_config = {
        'max_features': max_features,
        'sequence_length': sequence_length,
        'embedding_dim': embedding_dim,
        'epochs': 10,
        'batch_size': 32
    }
    
    # EXPERIMENT 1: Number of RNN Layers 
    print("\n" + "="*50)
    print("EXPERIMENT 1: TESTING NUMBER OF RNN LAYERS")
    print("="*50)
    
    layer_experiments = [
        {'name': '1_layer', 'num_rnn_layers': 1, 'rnn_units': 64, 'bidirectional': False},
        {'name': '2_layers', 'num_rnn_layers': 2, 'rnn_units': 64, 'bidirectional': False},
        {'name': '3_layers', 'num_rnn_layers': 3, 'rnn_units': 64, 'bidirectional': False}
    ]
    
    for exp in layer_experiments:
        print(f"\n  Testing {exp['name']} - {exp['num_rnn_layers']} RNN Layer(s)...")
        
        config = {**base_config, **exp}
        model = create_rnn_model(config)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train_vec, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_val_vec, y_val),
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test_vec, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        # Store results
        results.append({
            'experiment': 'rnn_layers',
            'variant': exp['name'],
            'description': f"{exp['num_rnn_layers']} RNN Layer(s)",
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        training_histories[f"layers_{exp['name']}"] = history.history
        
        # Save model
        model_path = f"models/rnn_layers_{exp['name']}.h5"
        weights_path = f"weights/rnn_layers_{exp['name']}.weights.h5"
        
        model.save(model_path)
        model.save_weights(weights_path)
        
        saved_models[f"layers_{exp['name']}"] = {
            'model': model_path,
            'weights': weights_path,
            'f1_score': f1,
            'config': config
        }
        
        print(f"    Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"    Model saved: {model_path}")
    
    # EXPERIMENT 2: Number of RNN Cells 
    print("\n" + "="*50)
    print("EXPERIMENT 2: TESTING NUMBER OF RNN CELLS")
    print("="*50)
    
    cell_experiments = [
        {'name': '32_cells', 'num_rnn_layers': 2, 'rnn_units': 32, 'bidirectional': False},
        {'name': '64_cells', 'num_rnn_layers': 2, 'rnn_units': 64, 'bidirectional': False},
        {'name': '128_cells', 'num_rnn_layers': 2, 'rnn_units': 128, 'bidirectional': False}
    ]
    
    for exp in cell_experiments:
        print(f"\n  Testing {exp['name']} - {exp['rnn_units']} cells per layer...")
        
        config = {**base_config, **exp}
        model = create_rnn_model(config)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train_vec, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_val_vec, y_val),
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test_vec, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        # Store results
        results.append({
            'experiment': 'rnn_cells',
            'variant': exp['name'],
            'description': f"{exp['rnn_units']} cells per layer",
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        training_histories[f"cells_{exp['name']}"] = history.history
        
        # Save model
        model_path = f"models/rnn_cells_{exp['name']}.h5"
        weights_path = f"weights/rnn_cells_{exp['name']}.weights.h5"
        
        model.save(model_path)
        model.save_weights(weights_path)
        
        saved_models[f"cells_{exp['name']}"] = {
            'model': model_path,
            'weights': weights_path,
            'f1_score': f1,
            'config': config
        }
        
        print(f"    Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"    Model saved: {model_path}")
    
    # EXPERIMENT 3: Bidirectional vs Unidirectional
    print("\n" + "="*50)
    print("EXPERIMENT 3: BIDIRECTIONAL vs UNIDIRECTIONAL")
    print("="*50)
    
    direction_experiments = [
        {'name': 'unidirectional', 'num_rnn_layers': 2, 'rnn_units': 64, 'bidirectional': False},
        {'name': 'bidirectional', 'num_rnn_layers': 2, 'rnn_units': 64, 'bidirectional': True}
    ]
    
    for exp in direction_experiments:
        direction_type = "Bidirectional" if exp['bidirectional'] else "Unidirectional"
        print(f"\n  Testing {exp['name']} - {direction_type} RNN...")
        
        config = {**base_config, **exp}
        model = create_rnn_model(config)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train_vec, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_val_vec, y_val),
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test_vec, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        # Store results
        results.append({
            'experiment': 'rnn_direction',
            'variant': exp['name'],
            'description': direction_type,
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        training_histories[f"direction_{exp['name']}"] = history.history
        
        # Save model
        model_path = f"models/rnn_direction_{exp['name']}.h5"
        weights_path = f"weights/rnn_direction_{exp['name']}.weights.h5"
        
        model.save(model_path)
        model.save_weights(weights_path)
        
        saved_models[f"direction_{exp['name']}"] = {
            'model': model_path,
            'weights': weights_path,
            'f1_score': f1,
            'config': config
        }
        
        print(f"    Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"    Model saved: {model_path}")
    
    # FORWARD PROPAGATION FROM SCRATCH TESTING
    print("\n" + "="*60)
    print("TESTING FORWARD PROPAGATION FROM SCRATCH")
    print("="*60)
    
    test_forward_propagation_comparison(saved_models, X_test_vec, y_test)
    
    # PLOTTING RESULTS
    plot_training_histories(training_histories)
    
    # RESULTS SUMMARY
    print_results_summary(results, saved_models)
    
    return pd.DataFrame(results), saved_models

def test_forward_propagation_comparison(saved_models, X_test_vec, y_test):    
    print("\n--- Forward Propagation Comparison ---")
    
    X_test_small = X_test_vec
    y_test_small = y_test
    
    for model_key in saved_models:
        model_info = saved_models[model_key]
        model_path = model_info['model']
        
        if os.path.exists(model_path):
            print(f"\n🔍 Testing {model_key}...")
            
            try:
                # Load Keras model and predict
                keras_model = keras.models.load_model(model_path)
                keras_pred = keras_model.predict(X_test_small, verbose=0)
                keras_classes = np.argmax(keras_pred, axis=1)
                
                # Test from-scratch implementation
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
                
                print(f"    Prediction match: {matches}/{len(keras_classes)} ({match_percentage:.1f}%)")
                print(f"    Keras F1-Score: {keras_f1:.4f}")
                print(f"    Scratch F1-Score: {scratch_f1:.4f}")
                print(f"    F1 difference: {abs(keras_f1 - scratch_f1):.4f}")
                
                if match_percentage > 95:
                    print(f"    Excellent match! From-scratch implementation is correct.")
                elif match_percentage > 80:
                    print(f"    Good match! Minor differences likely due to numerical precision.")
                else:
                    print(f"     Significant differences detected. Check implementation.")
                
            except Exception as e:
                print(f"    Error testing {model_key}: {e}")

def plot_training_histories(training_histories):
    print("\n  Generating training history plots...")
    
    experiments = [
        ('layers', 'Number of RNN Layers'),
        ('cells', 'Number of RNN Cells'),
        ('direction', 'Bidirectional vs Unidirectional')
    ]
    
    for exp_prefix, title in experiments:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'RNN Experiment: {title}', fontsize=14, fontweight='bold')
        
        for key, history in training_histories.items():
            if key.startswith(exp_prefix):
                variant_name = key.replace(f'{exp_prefix}_', '').replace('_', ' ').title()
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
        plot_filename = f'plots/rnn_{exp_prefix}_training_history.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_filename}")
        plt.show()

def print_results_summary(results, saved_models):
    print("\n" + "="*60)
    print("RNN EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    # Summary by experiment
    for experiment in df_results['experiment'].unique():
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        
        print(f"\n  {experiment.upper().replace('_', ' ')} EXPERIMENT")
        print(f"   Best Variant: {best_variant['variant']} ({best_variant['description']})")
        print(f"   Best F1-Score: {best_variant['f1_score']:.4f}")
        print(f"   Best Accuracy: {best_variant['accuracy']:.4f}")
        
        # Print all variants for comparison
        print(f"   All Results:")
        for _, row in exp_data.iterrows():
            print(f"     • {row['variant']}: F1={row['f1_score']:.4f}, Acc={row['accuracy']:.4f}")
        
        # Conclusions
        if experiment == 'rnn_layers':
            print(f"     CONCLUSION: More RNN layers can capture complex patterns but may")
            print(f"      suffer from vanishing gradients and overfitting.")
        elif experiment == 'rnn_cells':
            print(f"     CONCLUSION: More cells increase model capacity but require")
            print(f"      more data and computational resources to train effectively.")
        elif experiment == 'rnn_direction':
            print(f"     CONCLUSION: Bidirectional RNNs capture both past and future")
            print(f"      context but double parameters and cannot process streaming data.")
    
    # Overall best model
    best_overall = df_results.loc[df_results['f1_score'].idxmax()]
    print(f"\n🏆 OVERALL BEST MODEL")
    print(f"   Experiment: {best_overall['experiment']}")
    print(f"   Variant: {best_overall['variant']}")
    print(f"   Description: {best_overall['description']}")
    print(f"   F1-Score: {best_overall['f1_score']:.4f}")
    print(f"   Accuracy: {best_overall['accuracy']:.4f}")
    
    # Models summary
    print(f"\n SAVED MODELS SUMMARY")
    print(f"   Total models saved: {len(saved_models)}")
    for model_key, info in saved_models.items():
        print(f"     {model_key}: F1={info['f1_score']:.4f} | {info['model']}")
    
    print(f"\n  Analysis Complete! All models and weights saved successfully.")

if __name__ == "__main__":
    print("  Starting RNN Hyperparameter Analysis...")
    results_df, models_dict = analyze_rnn_hyperparameters()
    print("\n  RNN Analysis Complete!")
