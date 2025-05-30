import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

# Import scratch implementations
from cnn_scratch import CNNFromScratch

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_cnn_hyperparameters(return_models=False):
    """CNN Hyperparameter Analysis with CIFAR-10"""
    
    # GPU optimization
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU acceleration for CNN training!")
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Load and prepare CIFAR-10 data with CORRECT split
    (x_train_full, y_train_full), (x_test_original, y_test_original) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test_original = x_test_original.astype('float32') / 255.0
    
    print("Original Dataset Info:")
    print(f"Train Dataset Length: {len(x_train_full)}")
    print(f"Test Dataset Length: {len(x_test_original)}")
    
    # CORRECT DATA SPLIT: 40k train, 10k validation from train, 10k test from original test
    np.random.seed(42)
    train_indices = np.random.choice(len(x_train_full), 40000, replace=False)
    remaining_indices = np.setdiff1d(np.arange(len(x_train_full)), train_indices)
    val_indices = np.random.choice(remaining_indices, 10000, replace=False)
    test_indices = np.random.choice(len(x_test_original), 10000, replace=False)
    
    # Create the splits
    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices].flatten()
    x_val = x_train_full[val_indices] 
    y_val = y_train_full[val_indices].flatten()
    x_test = x_test_original[test_indices]
    y_test = y_test_original[test_indices].flatten()
    
    print(f"\nCORRECTED Data Split:")
    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Validation set: {x_val.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    
    results = []
    training_histories = {}
    saved_models = {}
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    
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
    
    best_conv_f1 = 0
    best_conv_name = ""
    
    for variant in layer_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        model = keras.Sequential(variant['layers'])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
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
        
        # Track best for summary
        if f1 > best_conv_f1:
            best_conv_f1 = f1
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
        
        # SAVE EVERY MODEL (not just best)
        model_filename = f'models/conv_layers_{variant["name"]}.h5'
        weights_filename = f'weights/conv_layers_{variant["name"]}.weights.h5'
        
        model.save(model_filename)
        model.save_weights(weights_filename)
        
        saved_models[f'conv_layers_{variant["name"]}'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': f1,
            'experiment': 'conv_layers',
            'variant': variant['name']
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"  ✓ Model saved: {model_filename}")
    
    # ========== EXPERIMENT 2: Number of Filters per Layer ==========
    print("\n" + "="*60)
    print("EXPERIMENT 2: TESTING NUMBER OF FILTERS PER LAYER")
    print("="*60)
    
    filter_variants = [
        {'name': 'small_filters_16_32_64', 'description': 'Small Filters (16-32-64)', 'filters': [16, 32, 64]},
        {'name': 'medium_filters_32_64_128', 'description': 'Medium Filters (32-64-128)', 'filters': [32, 64, 128]},
        {'name': 'large_filters_64_128_256', 'description': 'Large Filters (64-128-256)', 'filters': [64, 128, 256]}
    ]
    
    best_filter_f1 = 0
    best_filter_name = ""
    
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
                     loss='sparse_categorical_crossentropy',
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
        
        # Track best for summary
        if f1 > best_filter_f1:
            best_filter_f1 = f1
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
        
        # SAVE EVERY MODEL (not just best)
        model_filename = f'models/filter_count_{variant["name"]}.h5'
        weights_filename = f'weights/filter_count_{variant["name"]}.weights.h5'
        
        model.save(model_filename)
        model.save_weights(weights_filename)
        
        saved_models[f'filter_count_{variant["name"]}'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': f1,
            'experiment': 'filter_count',
            'variant': variant['name']
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"  ✓ Model saved: {model_filename}")
    
    # ========== EXPERIMENT 3: Filter Size ==========
    print("\n" + "="*60)
    print("EXPERIMENT 3: TESTING FILTER SIZES")
    print("="*60)
    
    kernel_variants = [
        {'name': '3x3_kernels', 'description': '3x3 Kernels', 'kernel_size': (3, 3)},
        {'name': '5x5_kernels', 'description': '5x5 Kernels', 'kernel_size': (5, 5)},
        {'name': '7x7_kernels', 'description': '7x7 Kernels', 'kernel_size': (7, 7)}
    ]
    
    best_kernel_f1 = 0
    best_kernel_name = ""
    
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
                     loss='sparse_categorical_crossentropy',
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
        
        # Track best for summary
        if f1 > best_kernel_f1:
            best_kernel_f1 = f1
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
        
        # SAVE EVERY MODEL (not just best)
        model_filename = f'models/kernel_size_{variant["name"]}.h5'
        weights_filename = f'weights/kernel_size_{variant["name"]}.weights.h5'
        
        model.save(model_filename)
        model.save_weights(weights_filename)
        
        saved_models[f'kernel_size_{variant["name"]}'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': f1,
            'experiment': 'kernel_size',
            'variant': variant['name']
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"  ✓ Model saved: {model_filename}")
    
    # ========== EXPERIMENT 4: Pooling Layer Types ==========
    print("\n" + "="*60)
    print("EXPERIMENT 4: TESTING POOLING LAYER TYPES")
    print("="*60)
    
    pooling_variants = [
        {'name': 'max_pooling', 'description': 'Max Pooling'},
        {'name': 'average_pooling', 'description': 'Average Pooling'}
    ]
    
    best_pooling_f1 = 0
    best_pooling_name = ""
    
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
                     loss='sparse_categorical_crossentropy',
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
        
        # Track best for summary
        if f1 > best_pooling_f1:
            best_pooling_f1 = f1
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
        
        # SAVE EVERY MODEL (not just best)
        model_filename = f'models/pooling_type_{variant["name"]}.h5'
        weights_filename = f'weights/pooling_type_{variant["name"]}.weights.h5'
        
        model.save(model_filename)
        model.save_weights(weights_filename)
        
        saved_models[f'pooling_type_{variant["name"]}'] = {
            'model': model_filename,
            'weights': weights_filename,
            'f1_score': f1,
            'experiment': 'pooling_type',
            'variant': variant['name']
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"  ✓ Model saved: {model_filename}")
    
    # ========== FORWARD PROPAGATION FROM SCRATCH TESTING ==========
    print("\n" + "="*60)
    print("TESTING FORWARD PROPAGATION FROM SCRATCH")
    print("="*60)
    
    # Test CNN from scratch implementation
    test_forward_propagation_cnn(saved_models, x_test, y_test)
    
    # ========== PLOTTING ==========
    print("\n" + "="*60)
    print("PLOTTING TRAINING HISTORIES FOR EACH EXPERIMENT")
    print("="*60)
    
    os.makedirs('plots', exist_ok=True)
    
    # Plot for each experiment
    experiments = ['conv_layers', 'filter_count', 'kernel_size', 'pooling_type']
    experiment_titles = [
        'Number of Convolutional Layers',
        'Number of Filters per Layer', 
        'Filter Size',
        'Pooling Layer Types'
    ]
    
    for exp, title in zip(experiments, experiment_titles):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'CNN Experiment: {title} - Training History', fontsize=14)
        
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
        plt.savefig(f'plots/cnn_{exp}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
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
    
    print(f"\n" + "="*60)
    print("ALL SAVED MODELS SUMMARY")
    print("="*60)
    total_models = len(saved_models)
    print(f"Total models saved: {total_models}")
    
    for model_key, info in saved_models.items():
        print(f"✓ {model_key}: F1={info['f1_score']:.4f} | {info['model']}")
    
    print(f"\nCNN Hyperparameter Analysis Complete!")
    print(f"✓ All {total_models} model weights saved in weights/ directory")
    
    if return_models:
        return df_results, saved_models
    else:
        return df_results

def test_forward_propagation_cnn(saved_models, x_test, y_test):
    """Test CNN forward propagation from scratch vs Keras"""
    
    print("\n--- Testing CNN Forward Propagation From Scratch ---")
    
    # Use a smaller test set for comparison
    x_test_small = x_test[:100]
    y_test_small = y_test[:100]
    
    # Test a few representative models (not all to avoid too much output)
    test_models = ['conv_layers_2_conv_layers', 'filter_count_medium_filters_32_64_128', 'kernel_size_3x3_kernels']
    
    for model_key in test_models:
        if model_key in saved_models:
            model_info = saved_models[model_key]
            model_path = model_info['model']
            
            if os.path.exists(model_path):
                print(f"\nTesting {model_key} model...")
                
                try:
                    # Load Keras model
                    keras_model = keras.models.load_model(model_path)
                    keras_pred = keras_model.predict(x_test_small, verbose=0)
                    keras_classes = np.argmax(keras_pred, axis=1)
                    
                    # Test scratch implementation
                    cnn_scratch = CNNFromScratch()
                    cnn_scratch.load_keras_model(keras_model)
                    
                    scratch_pred = cnn_scratch.forward(x_test_small)
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
    
    print("\n🔥 Running CNN analysis with CIFAR-10...")
    cnn_results = analyze_cnn_hyperparameters()
    
    print("\nCNN Analysis Complete!")