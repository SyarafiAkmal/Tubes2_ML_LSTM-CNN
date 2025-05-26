import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from cnn.cnn_keras import run_cnn_experiments
from cnn.cnn_scratch import test_cnn_scratch
from rnn.rnn_keras import run_rnn_experiments  
from rnn.rnn_scratch import test_rnn_scratch
from lstm.lstm_keras import run_lstm_experiments
from lstm.lstm_scratch import test_lstm_scratch

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'plots', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Created necessary directories")

def run_cnn_only():
    """Run only CNN experiments"""
    print("="*40)
    print("RUNNING CNN EXPERIMENTS ONLY")
    print("="*40)
    
    try:
        cnn_results = run_cnn_experiments()
        
        print("\nTesting CNN from scratch implementation...")
        cnn_scratch = test_cnn_scratch()
        
        return {'CNN': cnn_results}
        
    except Exception as e:
        print(f"Error in CNN experiments: {e}")
        return {}

def run_rnn_only():
    """Run only RNN experiments"""
    print("="*40)
    print("RUNNING RNN EXPERIMENTS ONLY")
    print("="*40)
    
    try:
        rnn_results = run_rnn_experiments()
        
        print("\nTesting RNN from scratch implementation...")
        rnn_scratch = test_rnn_scratch()
        
        return {'RNN': rnn_results}
        
    except Exception as e:
        print(f"Error in RNN experiments: {e}")
        return {}

def run_lstm_only():
    """Run only LSTM experiments"""
    print("="*40)
    print("RUNNING LSTM EXPERIMENTS ONLY")
    print("="*40)
    
    try:
        lstm_results = run_lstm_experiments()
        
        print("\nTesting LSTM from scratch implementation...")
        lstm_scratch = test_lstm_scratch()
        
        return {'LSTM': lstm_results}
        
    except Exception as e:
        print(f"Error in LSTM experiments: {e}")
        return {}

def run_all_experiments():
    """Run all experiments for CNN, RNN, and LSTM"""
    
    print("="*60)
    print("TUGAS BESAR 2 IF3270 - CNN AND RNN IMPLEMENTATION")
    print("="*60)
    
    all_results = {}
    
    cnn_results = run_cnn_only()
    rnn_results = run_rnn_only()
    lstm_results = run_lstm_only()
    
    all_results.update(cnn_results)
    all_results.update(rnn_results)
    all_results.update(lstm_results)
    
    return all_results

def save_results(results):
    """Save experiment results to file"""
    import json
    
    json_results = {}
    for model_type, experiments in results.items():
        json_results[model_type] = {}
        for exp_name, score in experiments.items():
            json_results[model_type][exp_name] = float(score)
    
    # Save to JSON 
    with open('results/experiment_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to results/experiment_results.json")

def plot_comparison_results(results):
    """Create comparison plots for all experiments"""
    
    if not results:
        print("No results to plot")
        return
    
    num_models = len(results)
    if num_models == 0:
        return
    
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 6))
    if num_models == 1:
        axes = [axes]
    
    for i, (model_type, model_data) in enumerate(results.items()):
        experiments = list(model_data.keys())
        scores = list(model_data.values())
        
        axes[i].bar(range(len(experiments)), scores)
        axes[i].set_title(f'{model_type} Experiment Results')
        axes[i].set_ylabel('F1-Score')
        axes[i].set_xticks(range(len(experiments)))
        axes[i].set_xticklabels(experiments, rotation=45, ha='right')
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('plots/experiment_comparison.jpg', dpi=300, bbox_inches='tight', format='jpg')
    plt.show()
    
    print("Comparison plot saved to plots/experiment_comparison.jpg")

def print_results_summary(results):
    """Print summary of results"""
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    for model_type, experiments in results.items():
        print(f"\n{model_type} Results:")
        print("-" * 30)
        for experiment, f1_score in experiments.items():
            print(f"{experiment}: F1-Score = {f1_score:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run ML experiments for Tugas Besar 2 IF3270')
    parser.add_argument('--model', choices=['cnn', 'rnn', 'lstm', 'all'], default='all',
                       help='Choose which model to run (default: all)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    if args.model == 'cnn':
        results = run_cnn_only()
    elif args.model == 'rnn':
        results = run_rnn_only()
    elif args.model == 'lstm':
        results = run_lstm_only()
    else:  # all
        results = run_all_experiments()
    
    print_results_summary(results)
    
    # Save 
    if results:
        save_results(results)
        
        # Create plots
        if not args.no_plots:
            plot_comparison_results(results)
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED!")
    print("="*60)
    print("\nCheck the following directories for results:")
    print("- models/: Trained Keras models")
    print("- results/: Experiment results in JSON format")
    print("- plots/: Comparison plots (JPG format)")
    print("- data/: Cached datasets")

if __name__ == "__main__":
    main()
