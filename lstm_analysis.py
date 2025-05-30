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

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import custom modules
from utils.dataset_loader import DatasetLoader

def analyze_lstm_hyperparameters(return_models=False):
    """LSTM Hyperparameter Analysis with NusaX-Sentiment Dataset"""
    
    print("\n" + "="*60)
    print("LSTM HYPERPARAMETER ANALYSIS - NUSAX-SENTIMENT")
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
    
    # ========== EXPERIMENT 1: Number of LSTM Layers ==========
    print("\n" + "="*50)
    print("EXPERIMENT 1: TESTING NUMBER OF LSTM LAYERS")
    print("="*50)
    
    layer_variants = [
        {'name': '1_lstm_layer', 'description': '1 LSTM Layer', 'num_layers': 1},
        {'name': '2_lstm_layers', 'description': '2 LSTM Layers', 'num_layers': 2},
        {'name': '3_lstm_layers', 'description': '3 LSTM Layers', 'num_layers': 3}
    ]
    
    best_layer_model = None
    best_layer_f1 = 0
    best_layer_name = ""
    
    for variant in layer_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential()
        model.add(layers.Embedding(max_features, 128, input_length=sequence_length))
        
        # Add LSTM layers
        for i in range(variant['num_layers']):
            if i < variant['num_layers'] - 1:  # All layers except the last one return sequences
                model.add(layers.LSTM(64, return_sequences=True))
                model.add(layers.Dropout(0.3))
            else:  # Last layer doesn't return sequences
                model.add(layers.LSTM(64))
        
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(3, activation='softmax'))  # 3 classes
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        history = model.fit(X_train_vec, y_train,
                          epochs=15,
                          batch_size=32,
                          validation_data=(X_val_vec, y_val),
                          verbose=1)
        
        training_histories[f"lstm_layers_{variant['name']}"] = history.history
        
        # Evaluate on test set
        y_pred = model.predict(X_test_vec, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        if f1 > best_layer_f1:
            best_layer_f1 = f1
            best_layer_model = model
            best_layer_name = variant['name']
        
        results.append({
            'experiment': 'lstm_layers',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Save best layer model
    if best_layer_model is not None:
        os.makedirs('models', exist_ok=True)
        os.makedirs('weights', exist_ok=True)
        
        model_filename = f'models/best_lstm_layers_{best_layer_name}.h5'
        weights_filename = f'weights/best_lstm_layers_{best_layer_name}_weights.h5'
        
        best_layer_model.save(model_filename)
        best_layer_model.save_weights(weights_filename)
        
        saved_models['lstm_layers'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': best_layer_f1
        }
        print(f"\n✓ Best LSTM layers model saved: {model_filename}")
    
    # ========== EXPERIMENT 2: Number of Cells per Layer ==========
    print("\n" + "="*50)
    print("EXPERIMENT 2: TESTING NUMBER OF CELLS PER LAYER")
    print("="*50)
    
    cell_variants = [
        {'name': '32_cells', 'description': '32 Cells per Layer', 'cells': 32},
        {'name': '64_cells', 'description': '64 Cells per Layer', 'cells': 64},
        {'name': '128_cells', 'description': '128 Cells per Layer', 'cells': 128}
    ]
    
    best_cell_model = None
    best_cell_f1 = 0
    best_cell_name = ""
    
    for variant in cell_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential([
            layers.Embedding(max_features, 128, input_length=sequence_length),
            layers.LSTM(variant['cells'], return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(variant['cells']),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        history = model.fit(X_train_vec, y_train,
                          epochs=15,
                          batch_size=32,
                          validation_data=(X_val_vec, y_val),
                          verbose=1)
        
        training_histories[f"lstm_cells_{variant['name']}"] = history.history
        
        y_pred = model.predict(X_test_vec, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        if f1 > best_cell_f1:
            best_cell_f1 = f1
            best_cell_model = model
            best_cell_name = variant['name']
        
        results.append({
            'experiment': 'lstm_cells',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Save best cell model
    if best_cell_model is not None:
        model_filename = f'models/best_lstm_cells_{best_cell_name}.h5'
        weights_filename = f'weights/best_lstm_cells_{best_cell_name}_weights.h5'
        
        best_cell_model.save(model_filename)
        best_cell_model.save_weights(weights_filename)
        
        saved_models['lstm_cells'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': best_cell_f1
        }
        print(f"\n✓ Best LSTM cells model saved: {model_filename}")
    
    # ========== EXPERIMENT 3: Bidirectional vs Unidirectional ==========
    print("\n" + "="*50)
    print("EXPERIMENT 3: TESTING BIDIRECTIONAL vs UNIDIRECTIONAL")
    print("="*50)
    
    direction_variants = [
        {'name': 'unidirectional', 'description': 'Unidirectional LSTM'},
        {'name': 'bidirectional', 'description': 'Bidirectional LSTM'}
    ]
    
    best_direction_model = None
    best_direction_f1 = 0
    best_direction_name = ""
    
    for variant in direction_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential()
        model.add(layers.Embedding(max_features, 128, input_length=sequence_length))
        
        if variant['name'] == 'bidirectional':
            model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
            model.add(layers.Dropout(0.3))
            model.add(layers.Bidirectional(layers.LSTM(64)))
        else:
            model.add(layers.LSTM(64, return_sequences=True))
            model.add(layers.Dropout(0.3))
            model.add(layers.LSTM(64))
        
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(3, activation='softmax'))
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        history = model.fit(X_train_vec, y_train,
                          epochs=15,
                          batch_size=32,
                          validation_data=(X_val_vec, y_val),
                          verbose=1)
        
        training_histories[f"lstm_direction_{variant['name']}"] = history.history
        
        y_pred = model.predict(X_test_vec, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        
        if f1 > best_direction_f1:
            best_direction_f1 = f1
            best_direction_model = model
            best_direction_name = variant['name']
        
        results.append({
            'experiment': 'lstm_direction',
            'variant': variant['name'],
            'description': variant['description'],
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Save best direction model
    if best_direction_model is not None:
        model_filename = f'models/best_lstm_direction_{best_direction_name}.h5'
        weights_filename = f'weights/best_lstm_direction_{best_direction_name}_weights.h5'
        
        best_direction_model.save(model_filename)
        best_direction_model.save_weights(weights_filename)
        
        saved_models['lstm_direction'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': best_direction_f1
        }
        print(f"\n✓ Best LSTM direction model saved: {model_filename}")
    
    # ========== PLOTTING ==========
    print("\n" + "="*50)
    print("PLOTTING LSTM TRAINING HISTORIES")
    print("="*50)
    
    experiments = ['lstm_layers', 'lstm_cells', 'lstm_direction']
    experiment_titles = [
        'Number of LSTM Layers',
        'Number of Cells per Layer',
        'Bidirectional vs Unidirectional'
    ]
    
    for exp, title in zip(experiments, experiment_titles):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'LSTM Experiment: {title} - Training History', fontsize=14)
        
        for key, history in training_histories.items():
            if key.startswith(exp):
                variant_name = key.replace(f"{exp}_", "").replace("_", " ").title()
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
        plt.savefig(f'plots/lstm_{exp}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Print results summary
    print("\n" + "="*50)
    print("LSTM EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    
    df_results = pd.DataFrame(results)
    
    for experiment in df_results['experiment'].unique():
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        
        print(f"\n{experiment.upper()} - Best Variant: {best_variant['variant']}")
        print(f"  Description: {best_variant['description']}")
        print(f"  F1-Score: {best_variant['f1_score']:.4f}")
        print(f"  Accuracy: {best_variant['accuracy']:.4f}")
        
        # Print conclusions
        if experiment == 'lstm_layers':
            print(f"  📊 CONCLUSION - LSTM Layers Impact:")
            print(f"     Adding more LSTM layers generally improves feature extraction")
            print(f"     but may lead to overfitting with limited data.")
        elif experiment == 'lstm_cells':
            print(f"  📊 CONCLUSION - LSTM Cells Impact:")
            print(f"     More cells increase model capacity but require more data")
            print(f"     to avoid overfitting and increase computational cost.")
        elif experiment == 'lstm_direction':
            print(f"  📊 CONCLUSION - LSTM Direction Impact:")
            print(f"     Bidirectional LSTMs capture context from both directions")
            print(f"     but double the parameters and computational cost.")
    
    print("\n" + "="*50)
    print("WEIGHT SAVING SUMMARY")
    print("="*50)
    for exp, info in saved_models.items():
        print(f"✓ {exp}: {info['model']} (F1: {info['f1_score']:.4f})")
    
    print(f"\nLSTM Hyperparameter Analysis Complete!")
    print(f"✓ All model weights saved in weights/ directory")
    print(f"✓ Training history plots saved in plots/ directory")
    
    if return_models:
        return results, saved_models
    else:
        return df_results

if __name__ == "__main__":
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("\n🔥 Running LSTM analysis with NusaX-Sentiment...")
    lstm_results = analyze_lstm_hyperparameters()
    
    print("\nLSTM Analysis Complete!")