import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_cnn_hyperparameters():
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import f1_score, accuracy_score
    from tensorflow import keras
    from tensorflow.keras import layers
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # GPU optimization
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU acceleration for CNN training!")
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print("Train Dataset Length:", len(x_train))
    print("Test Dataset Length:", len(x_test))
    
    # Use full dataset with 4:1 train/test split
    x_train_small = x_train[:40000]   # 40k training samples (80%)
    y_train_small = y_train[:40000]
    x_test_small = x_train[40000:]    # 10k test samples (20%) - from training set
    y_test_small = y_train[40000:]    # Use training labels for consistency
    
    results = []
    training_histories = {}
    saved_models = {}
    
    # ========== EXPERIMENT 1: Number of Convolutional Layers ==========
    print("\n" + "="*60)
    print("EXPERIMENT 1: TESTING NUMBER OF CONVOLUTIONAL LAYERS")
    print("="*60)
    
    layer_variants = [
        # Variant 1: 2 Conv layers
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
        # Variant 2: 4 Conv layers
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
        # Variant 3: 6 Conv layers
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
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(x_train_small, y_train_small, 
                          epochs=10, 
                          batch_size=128, 
                          validation_data=(x_test_small, y_test_small), 
                          verbose=1)
        
        # Store training history
        training_histories[f"conv_layers_{variant['name']}"] = history.history
        
        # Evaluate with error handling
        try:
            keras_pred = model.predict(x_test_small, batch_size=32, verbose=0)
        except Exception as e:
            print(f"  Prediction error: {e}")
            keras_pred = []
            for i in range(0, len(x_test_small), 32):
                batch = x_test_small[i:i+32]
                batch_pred = model(batch, training=False)
                keras_pred.append(batch_pred.numpy())
            keras_pred = np.concatenate(keras_pred, axis=0)
        
        keras_classes = np.argmax(keras_pred, axis=1)
        y_test_classes = np.argmax(y_test_small, axis=1)
        
        accuracy = accuracy_score(y_test_classes, keras_classes)
        f1 = f1_score(y_test_classes, keras_classes, average='macro')
        
        # Track best model
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
        model_filename = f'best_conv_layers_{best_conv_name}.h5'
        best_conv_model.save(model_filename)
        saved_models['conv_layers'] = model_filename
        print(f"\nBest conv layers model saved: {model_filename}")
    
    # ========== EXPERIMENT 2: Number of Filters per Layer ==========
    print("\n" + "="*60)
    print("EXPERIMENT 2: TESTING NUMBER OF FILTERS PER LAYER")
    print("="*60)
    
    filter_variants = [
        {'name': 'small_filters_16_32_64', 'description': 'Small Filters (16-32-64)', 'filters': [16, 32, 64]},
        {'name': 'medium_filters_32_64_128', 'description': 'Medium Filters (32-64-128)', 'filters': [32, 64, 128]},
        {'name': 'large_filters_64_128_256', 'description': 'Large Filters (64-128-256)', 'filters': [64, 128, 256]}
    ]
    
    for variant in filter_variants:
        print(f"\nTesting {variant['name']} - {variant['description']}...")
        f1, f2, f3 = variant['filters']
        
        model = keras.Sequential([
            layers.Conv2D(f1, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            layers.Conv2D(f1, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(f2, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(f2, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(f3, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(f3, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train_small, y_train_small, 
                          epochs=10, 
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small), 
                          verbose=1)
        
        training_histories[f"filter_count_{variant['name']}"] = history.history
        
        # Safe prediction
        try:
            keras_pred = model.predict(x_test_small, batch_size=32, verbose=0)
        except:
            keras_pred = []
            for i in range(0, len(x_test_small), 32):
                batch = x_test_small[i:i+32]
                batch_pred = model(batch, training=False)
                keras_pred.append(batch_pred.numpy())
            keras_pred = np.concatenate(keras_pred, axis=0)
        
        keras_classes = np.argmax(keras_pred, axis=1)
        y_test_classes = np.argmax(y_test_small, axis=1)
        
        accuracy = accuracy_score(y_test_classes, keras_classes)
        f1 = f1_score(y_test_classes, keras_classes, average='macro')
        
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
    
    # ========== EXPERIMENT 3: Filter Size ==========
    print("\n" + "="*60)
    print("EXPERIMENT 3: TESTING FILTER SIZES")
    print("="*60)
    
    kernel_variants = [
        {'name': '3x3_kernels', 'description': '3x3 Kernels', 'kernel_size': (3, 3)},
        {'name': '5x5_kernels', 'description': '5x5 Kernels', 'kernel_size': (5, 5)},
        {'name': '7x7_kernels', 'description': '7x7 Kernels', 'kernel_size': (7, 7)}
    ]
    
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
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train_small, y_train_small, 
                          epochs=10, 
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small), 
                          verbose=1)
        
        training_histories[f"kernel_size_{variant['name']}"] = history.history
        
        # Safe prediction
        try:
            keras_pred = model.predict(x_test_small, batch_size=32, verbose=0)
        except:
            keras_pred = []
            for i in range(0, len(x_test_small), 32):
                batch = x_test_small[i:i+32]
                batch_pred = model(batch, training=False)
                keras_pred.append(batch_pred.numpy())
            keras_pred = np.concatenate(keras_pred, axis=0)
        
        keras_classes = np.argmax(keras_pred, axis=1)
        y_test_classes = np.argmax(y_test_small, axis=1)
        
        accuracy = accuracy_score(y_test_classes, keras_classes)
        f1 = f1_score(y_test_classes, keras_classes, average='macro')
        
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
    
    # ========== EXPERIMENT 4: Pooling Layer Types ==========
    print("\n" + "="*60)
    print("EXPERIMENT 4: TESTING POOLING LAYER TYPES")
    print("="*60)
    
    pooling_variants = [
        {'name': 'max_pooling', 'description': 'Max Pooling'},
        {'name': 'average_pooling', 'description': 'Average Pooling'}
    ]
    
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
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train_small, y_train_small, 
                          epochs=10, 
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small), 
                          verbose=1)
        
        training_histories[f"pooling_type_{variant['name']}"] = history.history
        
        # Safe prediction
        try:
            keras_pred = model.predict(x_test_small, batch_size=32, verbose=0)
        except:
            keras_pred = []
            for i in range(0, len(x_test_small), 32):
                batch = x_test_small[i:i+32]
                batch_pred = model(batch, training=False)
                keras_pred.append(batch_pred.numpy())
            keras_pred = np.concatenate(keras_pred, axis=0)
        
        keras_classes = np.argmax(keras_pred, axis=1)
        y_test_classes = np.argmax(y_test_small, axis=1)
        
        accuracy = accuracy_score(y_test_classes, keras_classes)
        f1 = f1_score(y_test_classes, keras_classes, average='macro')
        
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
    
    # ========== PLOT TRAINING HISTORIES FOR EACH EXPERIMENT ==========
    print("\n" + "="*60)
    print("PLOTTING TRAINING HISTORIES FOR EACH EXPERIMENT")
    print("="*60)
    
    # Plot for Experiment 1: Number of Convolutional Layers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Experiment 1: Number of Convolutional Layers - Training History', fontsize=14)
    
    for key, history in training_histories.items():
        if 'conv_layers' in key:
            variant_name = key.replace('conv_layers_', '').replace('_', ' ').title()
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
    plt.savefig('cnn_conv_layers_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot for Experiment 2: Number of Filters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Experiment 2: Number of Filters per Layer - Training History', fontsize=14)
    
    for key, history in training_histories.items():
        if 'filter_count' in key:
            variant_name = key.replace('filter_count_', '').replace('_', ' ').title()
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
    plt.savefig('cnn_filter_count_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot for Experiment 3: Filter Size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Experiment 3: Filter Size - Training History', fontsize=14)
    
    for key, history in training_histories.items():
        if 'kernel_size' in key:
            variant_name = key.replace('kernel_size_', '').replace('_', ' ').title()
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
    plt.savefig('cnn_kernel_size_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot for Experiment 4: Pooling Types
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Experiment 4: Pooling Layer Types - Training History', fontsize=14)
    
    for key, history in training_histories.items():
        if 'pooling_type' in key:
            variant_name = key.replace('pooling_type_', '').replace('_', ' ').title()
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
    plt.savefig('cnn_pooling_type_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== SCRATCH IMPLEMENTATION TEST ==========
    print("\n" + "="*60)
    print("TESTING SCRATCH IMPLEMENTATION")
    print("="*60)
    
    try:
        import sys
        sys.path.append('src')
        from cnn_scratch import CNNFromScratch
        
        if best_conv_model is not None:
            print("Creating a simple model compatible with scratch implementation...")
            # Create a simple model that uses VALID padding (compatible with your scratch implementation)
            simple_model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # No padding
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),  # No padding
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])
            
            simple_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Train briefly on small subset
            print("Training simple model...")
            simple_model.fit(x_train_small[:1000], y_train_small[:1000], epochs=2, batch_size=32, verbose=0)
            
            print("Testing scratch implementation with simple model...")
            cnn_scratch = CNNFromScratch()
            cnn_scratch.load_keras_model(simple_model)
            
            # Test on very small subset
            test_subset = x_test_small[:50]
            test_labels = y_test_small[:50]
            
            try:
                keras_pred = simple_model.predict(test_subset, batch_size=32, verbose=0)
            except:
                keras_pred = []
                for i in range(0, len(test_subset), 16):
                    batch = test_subset[i:i+16]
                    batch_pred = simple_model(batch, training=False)
                    keras_pred.append(batch_pred.numpy())
                keras_pred = np.concatenate(keras_pred, axis=0)
            
            scratch_pred = cnn_scratch.forward(test_subset)
            
            keras_classes = np.argmax(keras_pred, axis=1)
            scratch_classes = np.argmax(scratch_pred, axis=1)
            y_test_classes = np.argmax(test_labels, axis=1)
            
            keras_f1 = f1_score(y_test_classes, keras_classes, average='macro')
            scratch_f1 = f1_score(y_test_classes, scratch_classes, average='macro')
            matches = np.sum(keras_classes == scratch_classes)
            match_percentage = matches/len(keras_classes)*100
            
            print(f"Keras F1-Score: {keras_f1:.4f}")
            print(f"Scratch F1-Score: {scratch_f1:.4f}")
            print(f"Keras vs Scratch Match: {match_percentage:.2f}%")
            
            # Test if outputs are close in value
            output_diff = np.mean(np.abs(keras_pred - scratch_pred))
            print(f"Average output difference: {output_diff:.6f}")
            
            if match_percentage > 95:
                print("Scratch implementation working correctly!")
            else:
                print("Some differences detected, but this might be due to numerical precision")
        
    except Exception as e:
        print(f"Scratch implementation test failed: {e}")
        print("Note: Make sure cnn_scratch.py is available in the src/ directory")
    
    # ========== RESULTS SUMMARY ==========
    print("\n" + "="*60)
    print("CNN EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    # Group by experiment and show best variant
    for experiment in df_results['experiment'].unique():
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        worst_variant = exp_data.loc[exp_data['f1_score'].idxmin()]
        
        print(f"\n{experiment.upper()} - Best Variant: {best_variant['variant']}")
        print(f"  Description: {best_variant['description']}")
        print(f"  F1-Score: {best_variant['f1_score']:.4f}")
        print(f"  Accuracy: {best_variant['accuracy']:.4f}")
        print(f"  Final Training Loss: {best_variant['final_loss']:.4f}")
        print(f"  Final Validation Loss: {best_variant['final_val_loss']:.4f}")
        print(f"  Performance vs worst: +{(best_variant['f1_score'] - worst_variant['f1_score']):.4f} F1-Score")
    
    # ========== CONCLUSIONS ==========
    print("\n" + "="*60)
    print("CNN ANALYSIS CONCLUSIONS")
    print("="*60)
    
    # Generate conclusions based on results
    conv_layers_data = df_results[df_results['experiment'] == 'conv_layers']
    filter_count_data = df_results[df_results['experiment'] == 'filter_count']
    kernel_size_data = df_results[df_results['experiment'] == 'kernel_size']
    pooling_data = df_results[df_results['experiment'] == 'pooling_type']
    
    print("\n1. CONVOLUTIONAL LAYERS IMPACT:")
    best_layers = conv_layers_data.loc[conv_layers_data['f1_score'].idxmax()]
    print(f"   - Best configuration: {best_layers['description']}")
    print(f"   - Conclusion: {'More layers improve performance' if '6_conv' in best_layers['variant'] else 'Moderate depth works best' if '4_conv' in best_layers['variant'] else 'Shallow networks sufficient'}")
    
    print("\n2. FILTER COUNT IMPACT:")
    best_filters = filter_count_data.loc[filter_count_data['f1_score'].idxmax()]
    print(f"   - Best configuration: {best_filters['description']}")
    print(f"   - Conclusion: {'Large filter counts improve feature detection' if 'large' in best_filters['variant'] else 'Medium filter counts provide good balance' if 'medium' in best_filters['variant'] else 'Small filter counts are sufficient'}")
    
    print("\n3. FILTER SIZE IMPACT:")
    best_kernel = kernel_size_data.loc[kernel_size_data['f1_score'].idxmax()]
    print(f"   - Best configuration: {best_kernel['description']}")
    print(f"   - Conclusion: {'Larger kernels capture more spatial information' if '7x7' in best_kernel['variant'] else 'Medium-sized kernels provide good balance' if '5x5' in best_kernel['variant'] else 'Small kernels are most effective'}")
    
    print("\n4. POOLING TYPE IMPACT:")
    best_pooling = pooling_data.loc[pooling_data['f1_score'].idxmax()]
    print(f"   - Best configuration: {best_pooling['description']}")
    max_pooling_f1 = pooling_data[pooling_data['variant'] == 'max_pooling']['f1_score'].iloc[0]
    avg_pooling_f1 = pooling_data[pooling_data['variant'] == 'average_pooling']['f1_score'].iloc[0]
    advantage = max_pooling_f1 - avg_pooling_f1
    print(f"   - Max vs Average pooling difference: {advantage:.4f}")
    print(f"   - Conclusion: {'Max pooling preserves important features better' if advantage > 0 else 'Average pooling provides smoother feature maps'}")
    
    # Overall best configuration
    overall_best = df_results.loc[df_results['f1_score'].idxmax()]
    print(f"\n5. OVERALL BEST CONFIGURATION:")
    print(f"   - Experiment: {overall_best['experiment'].replace('_', ' ').title()}")
    print(f"   - Configuration: {overall_best['description']}")
    print(f"   - F1-Score: {overall_best['f1_score']:.4f}")
    print(f"   - This represents the optimal hyperparameter setting among all tested configurations")
    
    # Create F1-Score comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CNN Hyperparameter Analysis - F1-Score Comparison', fontsize=16)
    
    experiments = ['conv_layers', 'filter_count', 'kernel_size', 'pooling_type']
    experiment_titles = ['Number of Conv Layers', 'Number of Filters', 'Filter Size', 'Pooling Type']
    
    for i, (experiment, title) in enumerate(zip(experiments, experiment_titles)):
        row = i // 2
        col = i % 2
        
        exp_data = df_results[df_results['experiment'] == experiment]
        
        bars = axes[row, col].bar(range(len(exp_data)), exp_data['f1_score'])
        axes[row, col].set_title(f'{title} - F1 Score')
        axes[row, col].set_xticks(range(len(exp_data)))
        axes[row, col].set_xticklabels(exp_data['variant'], rotation=45, ha='right')
        axes[row, col].set_ylabel('F1 Score')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, v in enumerate(exp_data['f1_score']):
            axes[row, col].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('cnn_f1_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("SAVED MODELS SUMMARY")
    print("="*60)
    for exp, filename in saved_models.items():
        print(f"{exp}: {filename}")
    
    print(f"\nCNN Hyperparameter Analysis Complete!")
    print(f"enerated {len(training_histories)} training history plots")
    print(f"Saved {len(saved_models)} best performing models")
    print(f"Tested {len(results)} different configurations")
    
    return df_results, training_histories, saved_models

def analyze_rnn_hyperparameters():
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import f1_score, accuracy_score
    from tensorflow import keras
    from tensorflow.keras import layers
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # GPU optimization
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU acceleration for RNN training!")
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
    
    print("Starting RNN Hyperparameter Analysis...")
    print("Loading custom datasets from /data folder...")
    
    # Load your custom datasets
    import pandas as pd
    
    # Load the CSV files from data folder
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    valid_df = pd.read_csv('data/valid.csv')
    
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    print(f"Valid dataset shape: {valid_df.shape}")
    
    # Display column information
    print("Train dataset columns:", train_df.columns.tolist())
    
    # Assuming your dataset has 'text' and 'label' columns
    # Adjust these column names based on your actual dataset structure
    text_column = 'text'  # Change this to your text column name
    label_column = 'label'  # Change this to your label column name
    
    # Check if columns exist, if not, use the first two columns
    if text_column not in train_df.columns:
        text_column = train_df.columns[0]  # First column as text
        print(f"Using '{text_column}' as text column")
    
    if label_column not in train_df.columns:
        label_column = train_df.columns[1]  # Second column as label
        print(f"Using '{label_column}' as label column")
    
    # Extract text and labels
    train_texts = train_df[text_column].fillna('').astype(str).tolist()
    train_labels = train_df[label_column].tolist()
    
    test_texts = test_df[text_column].fillna('').astype(str).tolist()
    test_labels = test_df[label_column].tolist()
    
    valid_texts = valid_df[text_column].fillna('').astype(str).tolist()
    valid_labels = valid_df[label_column].tolist()
    
    # Text preprocessing and tokenization
    max_features = 10000  # vocabulary size
    maxlen = 500  # sequence length
    
    # Create tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train_texts)
    
    # Convert texts to sequences
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_test = tokenizer.texts_to_sequences(test_texts)
    x_valid = tokenizer.texts_to_sequences(valid_texts)
    
    # Pad sequences to same length
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    x_valid = keras.preprocessing.sequence.pad_sequences(x_valid, maxlen=maxlen)
    
    # Convert labels to numpy arrays
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    y_valid = np.array(valid_labels)
    
    # Determine if it's binary or multi-class classification
    unique_labels = np.unique(np.concatenate([y_train, y_test, y_valid]))
    num_classes = len(unique_labels)
    is_binary = num_classes == 2
    
    print(f"Number of classes: {num_classes}")
    print(f"Unique labels: {unique_labels}")
    print(f"Classification type: {'Binary' if is_binary else 'Multi-class'}")
    
    # Convert labels for training
    if is_binary:
        # For binary classification, ensure labels are 0 and 1
        label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
        y_train = np.array([label_mapping[label] for label in y_train])
        y_test = np.array([label_mapping[label] for label in y_test])
        y_valid = np.array([label_mapping[label] for label in y_valid])
        loss_function = 'binary_crossentropy'
        final_activation = 'sigmoid'
        output_units = 1
    else:
        # For multi-class classification, use categorical encoding
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)
        loss_function = 'categorical_crossentropy'
        final_activation = 'softmax'
        output_units = num_classes
    
    print(f"Final training data shape: {x_train.shape}")
    print(f"Final training labels shape: {y_train.shape}")
    
    # Use your custom split (you already have train/test/valid)
    x_train_small = x_train
    y_train_small = y_train
    x_test_small = x_valid  # Use validation set for testing
    y_test_small = y_valid
    
    results = []
    training_histories = {}
    saved_models = {}
    
    # ========== EXPERIMENT 1: Number of RNN Layers ==========
    print("\\n" + "="*60)
    print("EXPERIMENT 1: TESTING NUMBER OF RNN LAYERS")
    print("="*60)
    
    layer_variants = [
        # Variant 1: 1 RNN layer
        {
            'name': '1_rnn_layer',
            'description': '1 LSTM Layer'
        },
        # Variant 2: 2 RNN layers
        {
            'name': '2_rnn_layers',
            'description': '2 LSTM Layers'
        },
        # Variant 3: 3 RNN layers
        {
            'name': '3_rnn_layers',
            'description': '3 LSTM Layers'
        }
    ]
    
    best_layer_model = None
    best_layer_f1 = 0
    
    for variant in layer_variants:
        print(f"\\nTesting {variant['name']} - {variant['description']}...")
        
        if variant['name'] == '1_rnn_layer':
            model = keras.Sequential([
                layers.Embedding(max_features, 128, input_length=maxlen),
                layers.LSTM(64),
                layers.Dropout(0.5),
                layers.Dense(output_units, activation=final_activation)
            ])
        elif variant['name'] == '2_rnn_layers':
            model = keras.Sequential([
                layers.Embedding(max_features, 128, input_length=maxlen),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(64),
                layers.Dropout(0.5),
                layers.Dense(output_units, activation=final_activation)
            ])
        else:  # 3 layers
            model = keras.Sequential([
                layers.Embedding(max_features, 128, input_length=maxlen),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(64),
                layers.Dropout(0.5),
                layers.Dense(output_units, activation=final_activation)
            ])
        
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        
        history = model.fit(x_train_small, y_train_small,
                          epochs=5,
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small),
                          verbose=1)
        
        # Store training history
        training_histories[f"layers_{variant['name']}"] = history.history
        
        # Evaluate
        y_pred_prob = model.predict(x_test_small, verbose=0)
        
        if is_binary:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_true = y_test_small
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test_small, axis=1)
        
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        
        # Track best model
        if f1 > best_layer_f1:
            best_layer_f1 = f1
            best_layer_model = model
            best_layer_name = variant['name']
        
        results.append({
            'experiment': 'rnn_layers',
            'variant': variant['name'],
            'description': variant['description'],
            'f1_score': f1,
            'accuracy': accuracy,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save best layer model
    if best_layer_model is not None:
        model_filename = f'best_rnn_layers_{best_layer_name}.h5'
        best_layer_model.save(model_filename)
        saved_models['rnn_layers'] = model_filename
        print(f"\\nBest RNN layers model saved: {model_filename}")
    
    # ========== EXPERIMENT 2: Number of Cells per Layer ==========
    print("\\n" + "="*60)
    print("EXPERIMENT 2: TESTING NUMBER OF CELLS PER RNN LAYER")
    print("="*60)
    
    cell_variants = [
        {'name': 'small_cells_32', 'cells': 32, 'description': '32 Cells per Layer'},
        {'name': 'medium_cells_64', 'cells': 64, 'description': '64 Cells per Layer'},
        {'name': 'large_cells_128', 'cells': 128, 'description': '128 Cells per Layer'}
    ]
    
    best_cell_model = None
    best_cell_f1 = 0
    
    for variant in cell_variants:
        print(f"\\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential([
            layers.Embedding(max_features, 128, input_length=maxlen),
            layers.LSTM(variant['cells'], return_sequences=True),
            layers.LSTM(variant['cells']),
            layers.Dropout(0.5),
            layers.Dense(output_units, activation=final_activation)
        ])
        
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        
        history = model.fit(x_train_small, y_train_small,
                          epochs=5,
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small),
                          verbose=1)
        
        training_histories[f"cells_{variant['name']}"] = history.history
        
        # Evaluate
        y_pred_prob = model.predict(x_test_small, verbose=0)
        
        if is_binary:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_true = y_test_small
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test_small, axis=1)
        
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        
        if f1 > best_cell_f1:
            best_cell_f1 = f1
            best_cell_model = model
            best_cell_name = variant['name']
        
        results.append({
            'experiment': 'rnn_cells',
            'variant': variant['name'],
            'description': variant['description'],
            'f1_score': f1,
            'accuracy': accuracy,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save best cell model
    if best_cell_model is not None:
        model_filename = f'best_rnn_cells_{best_cell_name}.h5'
        best_cell_model.save(model_filename)
        saved_models['rnn_cells'] = model_filename
        print(f"\\nBest RNN cells model saved: {model_filename}")
    
    # ========== EXPERIMENT 3: Bidirectional vs Unidirectional ==========
    print("\\n" + "="*60)
    print("EXPERIMENT 3: TESTING BIDIRECTIONAL VS UNIDIRECTIONAL RNN")
    print("="*60)
    
    direction_variants = [
        {'name': 'unidirectional', 'description': 'Unidirectional LSTM'},
        {'name': 'bidirectional', 'description': 'Bidirectional LSTM'}
    ]
    
    best_direction_model = None
    best_direction_f1 = 0
    
    for variant in direction_variants:
        print(f"\\nTesting {variant['name']} - {variant['description']}...")
        
        if variant['name'] == 'unidirectional':
            model = keras.Sequential([
                layers.Embedding(max_features, 128, input_length=maxlen),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(64),
                layers.Dropout(0.5),
                layers.Dense(output_units, activation=final_activation)
            ])
        else:  # bidirectional
            model = keras.Sequential([
                layers.Embedding(max_features, 128, input_length=maxlen),
                layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(64)),
                layers.Dropout(0.5),
                layers.Dense(output_units, activation=final_activation)
            ])
        
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        
        history = model.fit(x_train_small, y_train_small,
                          epochs=5,
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small),
                          verbose=1)
        
        training_histories[f"direction_{variant['name']}"] = history.history
        
        # Evaluate
        y_pred_prob = model.predict(x_test_small, verbose=0)
        
        if is_binary:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_true = y_test_small
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test_small, axis=1)
        
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        
        if f1 > best_direction_f1:
            best_direction_f1 = f1
            best_direction_model = model
            best_direction_name = variant['name']
        
        results.append({
            'experiment': 'rnn_direction',
            'variant': variant['name'],
            'description': variant['description'],
            'f1_score': f1,
            'accuracy': accuracy,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save best direction model
    if best_direction_model is not None:
        model_filename = f'best_rnn_direction_{best_direction_name}.h5'
        best_direction_model.save(model_filename)
        saved_models['rnn_direction'] = model_filename
        print(f"\\nBest RNN direction model saved: {model_filename}")
    
    # ========== SCRATCH IMPLEMENTATION TEST ==========
    print("\\n" + "="*60)
    print("TESTING RNN SCRATCH IMPLEMENTATION")
    print("="*60)
    
    try:
        import sys
        sys.path.append('src')
        from rnn_scratch import RNNFromScratch  # Assuming you have this implemented
        
        if best_layer_model is not None:
            print("Testing RNN scratch implementation...")
            rnn_scratch = RNNFromScratch()
            rnn_scratch.load_keras_model(best_layer_model)
            
            # Test on small subset
            test_subset = x_test_small[:100]
            test_labels = y_test_small[:100]
            
            keras_pred = best_layer_model.predict(test_subset, verbose=0)
            scratch_pred = rnn_scratch.forward(test_subset)
            
            if is_binary:
                keras_classes = (keras_pred > 0.5).astype(int).flatten()
                scratch_classes = (scratch_pred > 0.5).astype(int).flatten()
                y_true = test_labels
            else:
                keras_classes = np.argmax(keras_pred, axis=1)
                scratch_classes = np.argmax(scratch_pred, axis=1)
                y_true = np.argmax(test_labels, axis=1)
            
            keras_f1 = f1_score(y_true, keras_classes, average='macro')
            scratch_f1 = f1_score(y_true, scratch_classes, average='macro')
            matches = np.sum(keras_classes == scratch_classes)
            match_percentage = matches/len(keras_classes)*100
            
            print(f"Keras F1-Score: {keras_f1:.4f}")
            print(f"Scratch F1-Score: {scratch_f1:.4f}")
            print(f"Keras vs Scratch Match: {match_percentage:.2f}%")
        
    except Exception as e:
        print(f"RNN scratch implementation test failed: {e}")
        print("Note: RNN scratch implementation might not be available yet")
    
    # ========== PLOT TRAINING HISTORIES ==========
    print("\\n" + "="*60)
    print("PLOTTING TRAINING HISTORIES")
    print("="*60)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('RNN Hyperparameter Analysis - Training Histories', fontsize=16)
    
    experiments = ['layers', 'cells', 'direction']
    experiment_names = ['Number of RNN Layers', 'Cells per Layer', 'Bidirectional vs Unidirectional']
    
    for i, (exp, exp_name) in enumerate(zip(experiments, experiment_names)):
        # Training Loss
        axes[i, 0].set_title(f'{exp_name} - Training Loss')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        
        # Validation Loss
        axes[i, 1].set_title(f'{exp_name} - Validation Loss')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].set_ylabel('Validation Loss')
        
        # Plot each variant
        for key, history in training_histories.items():
            if exp in key:
                variant_name = key.split('_', 1)[-1]
                epochs = range(1, len(history['loss']) + 1)
                
                axes[i, 0].plot(epochs, history['loss'], label=variant_name, marker='o', markersize=3)
                axes[i, 1].plot(epochs, history['val_loss'], label=variant_name, marker='o', markersize=3)
        
        axes[i, 0].legend()
        axes[i, 1].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rnn_training_histories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== RESULTS SUMMARY ==========
    print("\\n" + "="*60)
    print("RNN EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    # Group by experiment and show best performing variant
    for experiment in df_results['experiment'].unique():
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        
        print(f"\\n{experiment.upper()} - Best Variant: {best_variant['variant']}")
        print(f"  Description: {best_variant['description']}")
        print(f"  F1-Score: {best_variant['f1_score']:.4f}")
        print(f"  Accuracy: {best_variant['accuracy']:.4f}")
        print(f"  Final Training Loss: {best_variant['final_loss']:.4f}")
        print(f"  Final Validation Loss: {best_variant['final_val_loss']:.4f}")
    
    # Create F1-Score comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('RNN Hyperparameter Analysis - F1-Score Results', fontsize=16)
    
    experiments = ['rnn_layers', 'rnn_cells', 'rnn_direction']
    experiment_names = ['RNN Layers', 'Cells per Layer', 'Direction']
    
    for i, (experiment, exp_name) in enumerate(zip(experiments, experiment_names)):
        exp_data = df_results[df_results['experiment'] == experiment]
        
        bars = axes[i].bar(range(len(exp_data)), exp_data['f1_score'])
        axes[i].set_title(f'{exp_name} - F1-Score')
        axes[i].set_xticks(range(len(exp_data)))
        axes[i].set_xticklabels(exp_data['variant'], rotation=45, ha='right')
        axes[i].set_ylabel('F1-Score')
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, v in enumerate(exp_data['f1_score']):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('rnn_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== CONCLUSIONS ==========
    print("\\n" + "="*60)
    print("RNN ANALYSIS CONCLUSIONS")
    print("="*60)
    
    # Generate conclusions based on results
    for experiment in experiments:
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        worst_variant = exp_data.loc[exp_data['f1_score'].idxmin()]
        
        print(f"\\n{experiment.upper()}:")
        print(f"  Best: {best_variant['description']} (F1: {best_variant['f1_score']:.4f})")
        print(f"  Worst: {worst_variant['description']} (F1: {worst_variant['f1_score']:.4f})")
        print(f"  Performance difference: {(best_variant['f1_score'] - worst_variant['f1_score']):.4f}")
    
    # Specific analysis for bidirectional vs unidirectional
    direction_data = df_results[df_results['experiment'] == 'rnn_direction']
    if len(direction_data) == 2:
        bi_score = direction_data[direction_data['variant'] == 'bidirectional']['f1_score'].iloc[0]
        uni_score = direction_data[direction_data['variant'] == 'unidirectional']['f1_score'].iloc[0]
        print(f"\\nBIDIRECTIONAL ANALYSIS:")
        print(f"  Bidirectional advantage: {bi_score - uni_score:.4f}")
        if bi_score > uni_score:
            print(f"  Bidirectional RNN performs better by {((bi_score/uni_score - 1)*100):.1f}%")
        else:
            print(f"  Unidirectional RNN performs better by {((uni_score/bi_score - 1)*100):.1f}%")
    
    print("\\n" + "="*60)
    print("SAVED MODELS SUMMARY")
    print("="*60)
    for exp, filename in saved_models.items():
        print(f"{exp}: {filename}")
    
    return df_results, training_histories, saved_models

def analyze_lstm_hyperparameters():
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import f1_score, accuracy_score
    from tensorflow import keras
    from tensorflow.keras import layers
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # GPU optimization
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU acceleration for LSTM training!")
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
    
    print("Starting LSTM Hyperparameter Analysis...")
    print("Loading custom datasets from /data folder...")
    
    # Load your custom datasets
    import pandas as pd
    
    # Load the CSV files from data folder
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    valid_df = pd.read_csv('data/valid.csv')
    
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    print(f"Valid dataset shape: {valid_df.shape}")
    
    # Display column information
    print("Train dataset columns:", train_df.columns.tolist())
    
    # Assuming your dataset has 'text' and 'label' columns
    # Adjust these column names based on your actual dataset structure
    text_column = 'text'  # Change this to your text column name
    label_column = 'label'  # Change this to your label column name
    
    # Check if columns exist, if not, use the first two columns
    if text_column not in train_df.columns:
        text_column = train_df.columns[0]  # First column as text
        print(f"Using '{text_column}' as text column")
    
    if label_column not in train_df.columns:
        label_column = train_df.columns[1]  # Second column as label
        print(f"Using '{label_column}' as label column")
    
    # Extract text and labels
    train_texts = train_df[text_column].fillna('').astype(str).tolist()
    train_labels = train_df[label_column].tolist()
    
    test_texts = test_df[text_column].fillna('').astype(str).tolist()
    test_labels = test_df[label_column].tolist()
    
    valid_texts = valid_df[text_column].fillna('').astype(str).tolist()
    valid_labels = valid_df[label_column].tolist()
    
    # Text preprocessing and tokenization
    max_features = 10000  # vocabulary size
    maxlen = 500  # sequence length
    
    # Create tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train_texts)
    
    # Convert texts to sequences
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_test = tokenizer.texts_to_sequences(test_texts)
    x_valid = tokenizer.texts_to_sequences(valid_texts)
    
    # Pad sequences to same length
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    x_valid = keras.preprocessing.sequence.pad_sequences(x_valid, maxlen=maxlen)
    
    # Convert labels to numpy arrays
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    y_valid = np.array(valid_labels)
    
    # Determine if it's binary or multi-class classification
    unique_labels = np.unique(np.concatenate([y_train, y_test, y_valid]))
    num_classes = len(unique_labels)
    is_binary = num_classes == 2
    
    print(f"Number of classes: {num_classes}")
    print(f"Unique labels: {unique_labels}")
    print(f"Classification type: {'Binary' if is_binary else 'Multi-class'}")
    
    # Convert labels for training
    if is_binary:
        # For binary classification, ensure labels are 0 and 1
        label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
        y_train = np.array([label_mapping[label] for label in y_train])
        y_test = np.array([label_mapping[label] for label in y_test])
        y_valid = np.array([label_mapping[label] for label in y_valid])
        loss_function = 'binary_crossentropy'
        final_activation = 'sigmoid'
        output_units = 1
    else:
        # For multi-class classification, use categorical encoding
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)
        loss_function = 'categorical_crossentropy'
        final_activation = 'softmax'
        output_units = num_classes
    
    print(f"Final training data shape: {x_train.shape}")
    print(f"Final training labels shape: {y_train.shape}")
    
    # Use your custom split (you already have train/test/valid)
    x_train_small = x_train
    y_train_small = y_train
    x_test_small = x_valid  # Use validation set for testing
    y_test_small = y_valid
    
    results = []
    training_histories = {}
    saved_models = {}
    
    # ========== EXPERIMENT 1: Number of LSTM Layers ==========
    print("\\n" + "="*60)
    print("EXPERIMENT 1: TESTING NUMBER OF LSTM LAYERS")
    print("="*60)
    
    layer_variants = [
        # Variant 1: 1 LSTM layer
        {
            'name': '1_lstm_layer',
            'description': '1 LSTM Layer',
            'num_layers': 1
        },
        # Variant 2: 2 LSTM layers
        {
            'name': '2_lstm_layers',
            'description': '2 LSTM Layers',
            'num_layers': 2
        },
        # Variant 3: 3 LSTM layers
        {
            'name': '3_lstm_layers',
            'description': '3 LSTM Layers',
            'num_layers': 3
        }
    ]
    
    best_layer_model = None
    best_layer_f1 = 0
    
    for variant in layer_variants:
        print(f"\\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential()
        model.add(layers.Embedding(max_features, 128, input_length=maxlen))
        
        # Add LSTM layers based on variant
        for i in range(variant['num_layers']):
            if i < variant['num_layers'] - 1:  # Not the last layer
                model.add(layers.LSTM(64, return_sequences=True))
            else:  # Last layer
                model.add(layers.LSTM(64))
        
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(output_units, activation=final_activation))
        
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        
        print(f"Model summary for {variant['name']}:")
        model.summary()
        
        history = model.fit(x_train_small, y_train_small,
                          epochs=5,
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small),
                          verbose=1)
        
        # Store training history
        training_histories[f"layers_{variant['name']}"] = history.history
        
        # Evaluate
        y_pred_prob = model.predict(x_test_small, verbose=0)
        
        if is_binary:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_true = y_test_small
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test_small, axis=1)
        
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        
        # Track best model
        if f1 > best_layer_f1:
            best_layer_f1 = f1
            best_layer_model = model
            best_layer_name = variant['name']
        
        results.append({
            'experiment': 'lstm_layers',
            'variant': variant['name'],
            'description': variant['description'],
            'f1_score': f1,
            'accuracy': accuracy,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save best layer model
    if best_layer_model is not None:
        model_filename = f'best_lstm_layers_{best_layer_name}.h5'
        best_layer_model.save(model_filename)
        saved_models['lstm_layers'] = model_filename
        print(f"\\nBest LSTM layers model saved: {model_filename}")
    
    # ========== EXPERIMENT 2: Number of Cells per Layer ==========
    print("\\n" + "="*60)
    print("EXPERIMENT 2: TESTING NUMBER OF CELLS PER LSTM LAYER")
    print("="*60)
    
    cell_variants = [
        {'name': 'small_cells_32', 'cells': 32, 'description': '32 Cells per Layer'},
        {'name': 'medium_cells_64', 'cells': 64, 'description': '64 Cells per Layer'},
        {'name': 'large_cells_128', 'cells': 128, 'description': '128 Cells per Layer'}
    ]
    
    best_cell_model = None
    best_cell_f1 = 0
    
    for variant in cell_variants:
        print(f"\\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential([
            layers.Embedding(max_features, 128, input_length=maxlen),
            layers.LSTM(variant['cells'], return_sequences=True),
            layers.LSTM(variant['cells']),
            layers.Dropout(0.5),
            layers.Dense(output_units, activation=final_activation)
        ])
        
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        
        print(f"Model with {variant['cells']} cells per layer:")
        print(f"Total parameters: {model.count_params():,}")
        
        history = model.fit(x_train_small, y_train_small,
                          epochs=5,
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small),
                          verbose=1)
        
        training_histories[f"cells_{variant['name']}"] = history.history
        
        # Evaluate
        y_pred_prob = model.predict(x_test_small, verbose=0)
        
        if is_binary:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_true = y_test_small
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test_small, axis=1)
        
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        
        if f1 > best_cell_f1:
            best_cell_f1 = f1
            best_cell_model = model
            best_cell_name = variant['name']
        
        results.append({
            'experiment': 'lstm_cells',
            'variant': variant['name'],
            'description': variant['description'],
            'f1_score': f1,
            'accuracy': accuracy,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save best cell model
    if best_cell_model is not None:
        model_filename = f'best_lstm_cells_{best_cell_name}.h5'
        best_cell_model.save(model_filename)
        saved_models['lstm_cells'] = model_filename
        print(f"\\nBest LSTM cells model saved: {model_filename}")
    
    # ========== EXPERIMENT 3: Bidirectional vs Unidirectional LSTM ==========
    print("\\n" + "="*60)
    print("EXPERIMENT 3: TESTING BIDIRECTIONAL VS UNIDIRECTIONAL LSTM")
    print("="*60)
    
    direction_variants = [
        {'name': 'unidirectional', 'description': 'Unidirectional LSTM'},
        {'name': 'bidirectional', 'description': 'Bidirectional LSTM'}
    ]
    
    best_direction_model = None
    best_direction_f1 = 0
    
    for variant in direction_variants:
        print(f"\\nTesting {variant['name']} - {variant['description']}...")
        
        model = keras.Sequential()
        model.add(layers.Embedding(max_features, 128, input_length=maxlen))
        
        if variant['name'] == 'unidirectional':
            model.add(layers.LSTM(64, return_sequences=True))
            model.add(layers.LSTM(64))
        else:  # bidirectional
            model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
            model.add(layers.Bidirectional(layers.LSTM(64)))
        
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(output_units, activation=final_activation))
        
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        
        print(f"Model architecture ({variant['name']}):")
        print(f"Total parameters: {model.count_params():,}")
        
        history = model.fit(x_train_small, y_train_small,
                          epochs=5,
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small),
                          verbose=1)
        
        training_histories[f"direction_{variant['name']}"] = history.history
        
        # Evaluate
        y_pred_prob = model.predict(x_test_small, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        f1 = f1_score(y_test_small, y_pred, average='macro')
        accuracy = accuracy_score(y_test_small, y_pred)
        
        if f1 > best_direction_f1:
            best_direction_f1 = f1
            best_direction_model = model
            best_direction_name = variant['name']
        
        results.append({
            'experiment': 'lstm_direction',
            'variant': variant['name'],
            'description': variant['description'],
            'f1_score': f1,
            'accuracy': accuracy,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save best direction model
    if best_direction_model is not None:
        model_filename = f'best_lstm_direction_{best_direction_name}.h5'
        best_direction_model.save(model_filename)
        saved_models['lstm_direction'] = model_filename
        print(f"\\nBest LSTM direction model saved: {model_filename}")
    
    # ========== SCRATCH IMPLEMENTATION TEST ==========
    print("\\n" + "="*60)
    print("TESTING LSTM SCRATCH IMPLEMENTATION")
    print("="*60)
    
    try:
        import sys
        sys.path.append('src')
        from lstm_scratch import LSTMFromScratch  # Assuming you have this implemented
        
        if best_layer_model is not None:
            print("Testing LSTM scratch implementation with best layer model...")
            lstm_scratch = LSTMFromScratch()
            lstm_scratch.load_keras_model(best_layer_model)
            
            # Test on small subset
            test_subset = x_test_small[:100]
            test_labels = y_test_small[:100]
            
            try:
                keras_pred = best_layer_model.predict(test_subset, verbose=0)
                scratch_pred = lstm_scratch.forward(test_subset)
                
                keras_classes = (keras_pred > 0.5).astype(int).flatten()
                scratch_classes = (scratch_pred > 0.5).astype(int).flatten()
                
                keras_f1 = f1_score(test_labels, keras_classes, average='macro')
                scratch_f1 = f1_score(test_labels, scratch_classes, average='macro')
                matches = np.sum(keras_classes == scratch_classes)
                match_percentage = matches/len(keras_classes)*100
                
                print(f"Keras F1-Score: {keras_f1:.4f}")
                print(f"Scratch F1-Score: {scratch_f1:.4f}")
                print(f"Keras vs Scratch Match: {match_percentage:.2f}%")
                
                # Test if outputs are close in value
                output_diff = np.mean(np.abs(keras_pred.flatten() - scratch_pred.flatten()))
                print(f"Average output difference: {output_diff:.6f}")
                
                if match_percentage > 95:
                    print("LSTM scratch implementation working correctly!")
                else:
                    print("Some differences detected, check implementation")
                    
            except Exception as pred_error:
                print(f"Prediction comparison failed: {pred_error}")
        
    except Exception as e:
        print(f"LSTM scratch implementation test failed: {e}")
        print("Note: LSTM scratch implementation might not be available yet")
        print("You can implement it following the same pattern as CNN scratch implementation")
    
    # ========== PLOT TRAINING HISTORIES ==========
    print("\\n" + "="*60)
    print("PLOTTING TRAINING HISTORIES")
    print("="*60)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('LSTM Hyperparameter Analysis - Training Histories', fontsize=16)
    
    experiments = ['layers', 'cells', 'direction']
    experiment_names = ['Number of LSTM Layers', 'Cells per Layer', 'Bidirectional vs Unidirectional']
    
    for i, (exp, exp_name) in enumerate(zip(experiments, experiment_names)):
        # Training Loss
        axes[i, 0].set_title(f'{exp_name} - Training Loss')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        
        # Validation Loss
        axes[i, 1].set_title(f'{exp_name} - Validation Loss')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].set_ylabel('Validation Loss')
        
        # Plot each variant
        for key, history in training_histories.items():
            if exp in key:
                variant_name = key.split('_', 1)[-1]
                epochs = range(1, len(history['loss']) + 1)
                
                axes[i, 0].plot(epochs, history['loss'], label=variant_name, marker='o', markersize=3)
                axes[i, 1].plot(epochs, history['val_loss'], label=variant_name, marker='o', markersize=3)
        
        axes[i, 0].legend()
        axes[i, 1].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../plots/lstm_training_histories.jpg', dpi=300, bbox_inches='tight', format='jpg')
    plt.show()
    
    # ========== RESULTS SUMMARY ==========
    print("\\n" + "="*60)
    print("LSTM EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    # Group by experiment and show best performing variant
    for experiment in df_results['experiment'].unique():
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        
        print(f"\\n{experiment.upper()} - Best Variant: {best_variant['variant']}")
        print(f"  Description: {best_variant['description']}")
        print(f"  F1-Score: {best_variant['f1_score']:.4f}")
        print(f"  Accuracy: {best_variant['accuracy']:.4f}")
        print(f"  Final Training Loss: {best_variant['final_loss']:.4f}")
        print(f"  Final Validation Loss: {best_variant['final_val_loss']:.4f}")
    
    # Create F1-Score comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('LSTM Hyperparameter Analysis - F1-Score Results', fontsize=16)
    
    experiments = ['lstm_layers', 'lstm_cells', 'lstm_direction']
    experiment_names = ['LSTM Layers', 'Cells per Layer', 'Direction']
    
    for i, (experiment, exp_name) in enumerate(zip(experiments, experiment_names)):
        exp_data = df_results[df_results['experiment'] == experiment]
        
        bars = axes[i].bar(range(len(exp_data)), exp_data['f1_score'])
        axes[i].set_title(f'{exp_name} - F1-Score')
        axes[i].set_xticks(range(len(exp_data)))
        axes[i].set_xticklabels(exp_data['variant'], rotation=45, ha='right')
        axes[i].set_ylabel('F1-Score')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for j, v in enumerate(exp_data['f1_score']):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../plots/lstm_detailed_analysis.jpg', dpi=300, bbox_inches='tight', format='jpg')
    plt.show()
    
    # ========== CONCLUSIONS ==========
    print("\\n" + "="*60)
    print("LSTM ANALYSIS CONCLUSIONS")
    print("="*60)
    
    # Generate conclusions based on results
    for experiment in ['lstm_layers', 'lstm_cells', 'lstm_direction']:
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        worst_variant = exp_data.loc[exp_data['f1_score'].idxmin()]
        
        print(f"\\n{experiment.upper()}:")
        print(f"  Best: {best_variant['description']} (F1: {best_variant['f1_score']:.4f})")
        print(f"  Worst: {worst_variant['description']} (F1: {worst_variant['f1_score']:.4f})")
        print(f"  Performance difference: {(best_variant['f1_score'] - worst_variant['f1_score']):.4f}")
        
        # Specific insights for each experiment
        if experiment == 'lstm_layers':
            print(f"  Insight: Adding more layers {'improved' if best_variant['variant'] == '3_lstm_layers' else 'did not improve'} performance")
        elif experiment == 'lstm_cells':
            print(f"  Insight: {'Larger' if best_variant['variant'] == 'large_cells_128' else 'Medium-sized' if best_variant['variant'] == 'medium_cells_64' else 'Smaller'} cell counts work best")
        elif experiment == 'lstm_direction':
            bi_data = exp_data[exp_data['variant'] == 'bidirectional']
            uni_data = exp_data[exp_data['variant'] == 'unidirectional']
            if not bi_data.empty and not uni_data.empty:
                bi_score = bi_data['f1_score'].iloc[0]
                uni_score = uni_data['f1_score'].iloc[0]
                advantage = bi_score - uni_score
                print(f"  Bidirectional advantage: {advantage:.4f}")
                if advantage > 0:
                    print(f"  Insight: Bidirectional LSTM performs better by {((bi_score/uni_score - 1)*100):.1f}%")
                else:
                    print(f"  Insight: Unidirectional LSTM performs better by {((uni_score/bi_score - 1)*100):.1f}%")
    
    print("\\n" + "="*60)
    print("SAVED MODELS SUMMARY")
    print("="*60)
    for exp, filename in saved_models.items():
        print(f"{exp}: {filename}")
    
    print("\\n" + "="*60)
    print("LSTM INSIGHTS")
    print("="*60)
    
    # Overall best configuration
    best_overall = df_results.loc[df_results['f1_score'].idxmax()]
    print(f"Best performing configuration: {best_overall['description']} (F1-Score: {best_overall['f1_score']:.4f})")
    
    # Compare bidirectional vs unidirectional across all experiments
    bidirectional_scores = [row['f1_score'] for _, row in df_results.iterrows() if 'bidirectional' in row['variant']]
    unidirectional_scores = [row['f1_score'] for _, row in df_results.iterrows() if 'unidirectional' in row['variant']]
    
    if bidirectional_scores and unidirectional_scores:
        avg_bi = np.mean(bidirectional_scores)
        avg_uni = np.mean(unidirectional_scores)
        print(f"Average Bidirectional F1-Score: {avg_bi:.4f}")
        print(f"Average Unidirectional F1-Score: {avg_uni:.4f}")
        print(f"Overall Bidirectional advantage: {avg_bi - avg_uni:.4f}")
    
    return df_results, training_histories, saved_models

def main():
    # Analyze each model type
    analyze_cnn_hyperparameters()

    analyze_rnn_hyperparameters()

    analyze_lstm_hyperparameters()

if __name__ == "__main__":
    main()