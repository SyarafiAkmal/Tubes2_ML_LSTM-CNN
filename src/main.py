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
    import numpy as np  # MISSING IMPORT - This was causing the error!
    import tensorflow as tf
    from sklearn.metrics import f1_score, accuracy_score
    from tensorflow import keras
    from tensorflow.keras import layers
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Fix TensorFlow prediction issue
    tf.config.run_functions_eagerly(False)
    
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
        print(f"\nTesting {variant['name']}...")
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
            print("  Trying alternative prediction method...")
            # Alternative prediction method
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
        print(f"\n✅ Best conv layers model saved: {model_filename}")
    
    # ========== EXPERIMENT 2: Number of Filters per Layer ==========
    print("\n" + "="*60)
    print("EXPERIMENT 2: TESTING NUMBER OF FILTERS PER LAYER")
    print("="*60)
    
    filter_variants = [
        {'name': 'small_filters_16_32_64', 'filters': [16, 32, 64]},
        {'name': 'medium_filters_32_64_128', 'filters': [32, 64, 128]},
        {'name': 'large_filters_64_128_256', 'filters': [64, 128, 256]}
    ]
    
    for variant in filter_variants:
        print(f"\nTesting {variant['name']}...")
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
        {'name': '3x3_kernels', 'kernel_size': (3, 3)},
        {'name': '5x5_kernels', 'kernel_size': (5, 5)},
        {'name': '7x7_kernels', 'kernel_size': (7, 7)}
    ]
    
    for variant in kernel_variants:
        print(f"\nTesting {variant['name']}...")
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
        {'name': 'max_pooling'},
        {'name': 'average_pooling'}
    ]
    
    for variant in pooling_variants:
        print(f"\nTesting {variant['name']}...")
        
        if variant['name'] == 'max_pooling':
            pooling_layer = layers.MaxPooling2D((2, 2))
        else:
            pooling_layer = layers.AveragePooling2D((2, 2))
        
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            pooling_layer,
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)) if variant['name'] == 'max_pooling' else layers.AveragePooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)) if variant['name'] == 'max_pooling' else layers.AveragePooling2D((2, 2)),
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
            'accuracy': accuracy,
            'f1_score': f1,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # ========== SCRATCH IMPLEMENTATION TEST ==========
    print("\n" + "="*60)
    print("TESTING SCRATCH IMPLEMENTATION")
    print("="*60)
    
    try:
        import sys
        sys.path.append('../src/cnn')
        from cnn_scratch import CNNFromScratch
        
        if best_conv_model is not None:
            print("Testing scratch implementation with best conv model...")
            cnn_scratch = CNNFromScratch()
            cnn_scratch.load_keras_model(best_conv_model)
            
            # Test on small subset
            test_subset = x_test_small[:100]
            test_labels = y_test_small[:100]
            
            keras_pred = best_conv_model.predict(test_subset, batch_size=32, verbose=0)
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
        
    except Exception as e:
        print(f"Scratch implementation test failed: {e}")
    
    # ========== RESULTS SUMMARY ==========
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    # Group by experiment and show best variant
    for experiment in df_results['experiment'].unique():
        exp_data = df_results[df_results['experiment'] == experiment]
        best_variant = exp_data.loc[exp_data['f1_score'].idxmax()]
        
        print(f"\n{experiment.upper()} - Best Variant: {best_variant['variant']}")
        print(f"  Accuracy: {best_variant['accuracy']:.4f}")
        print(f"  F1-Score: {best_variant['f1_score']:.4f}")
        print(f"  Final Loss: {best_variant['final_loss']:.4f}")
        print(f"  Final Val Loss: {best_variant['final_val_loss']:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CNN Hyperparameter Analysis Results', fontsize=16)
    
    experiments = df_results['experiment'].unique()
    for i, experiment in enumerate(experiments):
        row = i // 2
        col = i % 2
        
        exp_data = df_results[df_results['experiment'] == experiment]
        
        axes[row, col].bar(range(len(exp_data)), exp_data['f1_score'])
        axes[row, col].set_title(f'{experiment.replace("_", " ").title()} - F1 Score')
        axes[row, col].set_xticks(range(len(exp_data)))
        axes[row, col].set_xticklabels(exp_data['variant'], rotation=45, ha='right')
        axes[row, col].set_ylabel('F1 Score')
        
        # Add value labels on bars
        for j, v in enumerate(exp_data['f1_score']):
            axes[row, col].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
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
        print("🚀 Using GPU acceleration for RNN training!")
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
    
    print("Starting RNN Hyperparameter Analysis...")
    print("Loading and preparing sequential data...")
    
    # For RNN, we'll use IMDB sentiment analysis dataset (sequential text data)
    # This is more appropriate for RNN than CIFAR-10 (which is image data)
    max_features = 10000  # vocabulary size
    maxlen = 500  # sequence length
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
    
    # Pad sequences to same length
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Use subset for faster experiments (4:1 split as requested)
    x_train_small = x_train[:20000]  # 20k training samples
    y_train_small = y_train[:20000]
    x_test_small = x_train[20000:]  # 5k test samples (from remaining training data)
    y_test_small = y_train[20000:]
    
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
                layers.Dense(1, activation='sigmoid')
            ])
        elif variant['name'] == '2_rnn_layers':
            model = keras.Sequential([
                layers.Embedding(max_features, 128, input_length=maxlen),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(64),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])
        else:  # 3 layers
            model = keras.Sequential([
                layers.Embedding(max_features, 128, input_length=maxlen),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(64),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(x_train_small, y_train_small,
                          epochs=5,
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small),
                          verbose=1)
        
        # Store training history
        training_histories[f"layers_{variant['name']}"] = history.history
        
        # Evaluate
        y_pred_prob = model.predict(x_test_small, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        f1 = f1_score(y_test_small, y_pred, average='macro')
        accuracy = accuracy_score(y_test_small, y_pred)
        
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
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(x_train_small, y_train_small,
                          epochs=5,
                          batch_size=128,
                          validation_data=(x_test_small, y_test_small),
                          verbose=1)
        
        training_histories[f"cells_{variant['name']}"] = history.history
        
        # Evaluate
        y_pred_prob = model.predict(x_test_small, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        f1 = f1_score(y_test_small, y_pred, average='macro')
        accuracy = accuracy_score(y_test_small, y_pred)
        
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
                layers.Dense(1, activation='sigmoid')
            ])
        else:  # bidirectional
            model = keras.Sequential([
                layers.Embedding(max_features, 128, input_length=maxlen),
                layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(64)),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
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
        sys.path.append('.')
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
            
            keras_classes = (keras_pred > 0.5).astype(int).flatten()
            scratch_classes = (scratch_pred > 0.5).astype(int).flatten()
            
            keras_f1 = f1_score(test_labels, keras_classes, average='macro')
            scratch_f1 = f1_score(test_labels, scratch_classes, average='macro')
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
        print("🚀 Using GPU acceleration for LSTM training!")
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
    
    print("Starting LSTM Hyperparameter Analysis...")
    print("Loading and preparing sequential data...")
    
    # Using IMDB sentiment analysis dataset (appropriate for LSTM)
    max_features = 10000  # vocabulary size
    maxlen = 500  # sequence length
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
    
    # Pad sequences to same length
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Use 4:1 split as requested
    x_train_small = x_train[:20000]  # 20k training samples (80%)
    y_train_small = y_train[:20000]
    x_test_small = x_train[20000:25000]  # 5k test samples (20%)
    y_test_small = y_train[20000:25000]
    
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
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
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
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        f1 = f1_score(y_test_small, y_pred, average='macro')
        accuracy = accuracy_score(y_test_small, y_pred)
        
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
        print(f"\\n✅ Best LSTM layers model saved: {model_filename}")
    
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
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
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
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        f1 = f1_score(y_test_small, y_pred, average='macro')
        accuracy = accuracy_score(y_test_small, y_pred)
        
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
        print(f"\\n✅ Best LSTM cells model saved: {model_filename}")
    
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
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
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
        print(f"\\n✅ Best LSTM direction model saved: {model_filename}")
    
    # ========== SCRATCH IMPLEMENTATION TEST ==========
    print("\\n" + "="*60)
    print("TESTING LSTM SCRATCH IMPLEMENTATION")
    print("="*60)
    
    try:
        import sys
        sys.path.append('.')
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
                    print("✅ LSTM scratch implementation working correctly!")
                else:
                    print("⚠️ Some differences detected, check implementation")
                    
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
