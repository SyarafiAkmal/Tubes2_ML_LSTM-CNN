"""
Jupyter notebook for detailed analysis of experiment results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load experiment results from JSON file"""
    try:
        with open('../results/experiment_results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print("Results file not found. Please run main.py first.")
        return None

def analyze_cnn_results(results):
    """Analyze CNN experiment results"""
    if 'CNN' not in results:
        print("CNN results not found")
        return
    
    cnn_results = results['CNN']
    
    print("="*50)
    print("CNN EXPERIMENT ANALYSIS")
    print("="*50)
    
    # Create DataFrame for easier analysis
    df_data = []
    for exp_name, score in cnn_results.items():
        if 'layers_' in exp_name:
            category = 'Number of Layers'
            value = exp_name.split('_')[-1]
        elif 'filters_' in exp_name:
            category = 'Filter Configuration'
            value = exp_name.split('_')[-1]
        elif 'kernels_' in exp_name:
            category = 'Kernel Size'
            value = exp_name.split('_')[-1]
        elif 'pooling_' in exp_name:
            category = 'Pooling Type'
            value = exp_name.split('_')[-1]
        else:
            category = 'Other'
            value = exp_name
        
        df_data.append({
            'Experiment': exp_name,
            'Category': category,
            'Value': value,
            'F1_Score': score
        })
    
    df = pd.DataFrame(df_data)
    
    # Plot results by category
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CNN Experiment Results Analysis', fontsize=16)
    
    categories = df['Category'].unique()
    for i, category in enumerate(categories):
        if i < 4:  # Only plot first 4 categories
            row = i // 2
            col = i % 2
            
            cat_data = df[df['Category'] == category]
            axes[row, col].bar(cat_data['Value'], cat_data['F1_Score'])
            axes[row, col].set_title(f'{category}')
            axes[row, col].set_ylabel('F1-Score')
            axes[row, col].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('../plots/cnn_detailed_analysis.jpg', dpi=300, bbox_inches='tight', format='jpg')
    plt.show()
    
    # Print insights
    print("\nCNN INSIGHTS:")
    print("-" * 30)
    
    # Best performing configuration
    best_exp = max(cnn_results.items(), key=lambda x: x[1])
    print(f"Best performing configuration: {best_exp[0]} (F1-Score: {best_exp[1]:.4f})")
    
    # Category analysis
    for category in categories:
        cat_data = df[df['Category'] == category]
        if len(cat_data) > 1:
            best_in_cat = cat_data.loc[cat_data['F1_Score'].idxmax()]
            print(f"Best {category}: {best_in_cat['Value']} (F1-Score: {best_in_cat['F1_Score']:.4f})")

def analyze_rnn_results(results):
    """Analyze RNN experiment results"""
    if 'RNN' not in results:
        print("RNN results not found")
        return
    
    rnn_results = results['RNN']
    
    print("\n" + "="*50)
    print("RNN EXPERIMENT ANALYSIS")
    print("="*50)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    experiments = list(rnn_results.keys())
    scores = list(rnn_results.values())
    
    bars = ax.bar(range(len(experiments)), scores)
    ax.set_title('RNN Experiment Results')
    ax.set_ylabel('F1-Score')
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../plots/rnn_detailed_analysis.jpg', dpi=300, bbox_inches='tight', format='jpg')
    plt.show()
    
    # Print insights
    print("\nRNN INSIGHTS:")
    print("-" * 30)
    
    best_exp = max(rnn_results.items(), key=lambda x: x[1])
    print(f"Best performing configuration: {best_exp[0]} (F1-Score: {best_exp[1]:.4f})")
    
    # Compare bidirectional vs unidirectional
    bidirectional_scores = [score for exp, score in rnn_results.items() if 'bidirectional' in exp]
    unidirectional_scores = [score for exp, score in rnn_results.items() if 'unidirectional' in exp]
    
    if bidirectional_scores and unidirectional_scores:
        avg_bi = np.mean(bidirectional_scores)
        avg_uni = np.mean(unidirectional_scores)
        print(f"Average Bidirectional F1-Score: {avg_bi:.4f}")
        print(f"Average Unidirectional F1-Score: {avg_uni:.4f}")
        print(f"Bidirectional advantage: {avg_bi - avg_uni:.4f}")

def analyze_lstm_results(results):
    """Analyze LSTM experiment results"""
    if 'LSTM' not in results:
        print("LSTM results not found")
        return
    
    lstm_results = results['LSTM']
    
    print("\n" + "="*50)
    print("LSTM EXPERIMENT ANALYSIS")
    print("="*50)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    experiments = list(lstm_results.keys())
    scores = list(lstm_results.values())
    
    bars = ax.bar(range(len(experiments)), scores)
    ax.set_title('LSTM Experiment Results')
    ax.set_ylabel('F1-Score')
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../plots/lstm_detailed_analysis.jpg', dpi=300, bbox_inches='tight', format='jpg')
    plt.show()
    
    # Print insights
    print("\nLSTM INSIGHTS:")
    print("-" * 30)
    
    best_exp = max(lstm_results.items(), key=lambda x: x[1])
    print(f"Best performing configuration: {best_exp[0]} (F1-Score: {best_exp[1]:.4f})")
    
    # Compare bidirectional vs unidirectional
    bidirectional_scores = [score for exp, score in lstm_results.items() if 'bidirectional' in exp]
    unidirectional_scores = [score for exp, score in lstm_results.items() if 'unidirectional' in exp]
    
    if bidirectional_scores and unidirectional_scores:
        avg_bi = np.mean(bidirectional_scores)
        avg_uni = np.mean(unidirectional_scores)
        print(f"Average Bidirectional F1-Score: {avg_bi:.4f}")
        print(f"Average Unidirectional F1-Score: {avg_uni:.4f}")
        print(f"Bidirectional advantage: {avg_bi - avg_uni:.4f}")

def compare_all_models(results):
    """Compare performance across all models"""
    print("\n" + "="*50)
    print("OVERALL MODEL COMPARISON")
    print("="*50)
    
    # Get best score from each model type
    model_best_scores = {}
    for model_type, experiments in results.items():
        best_score = max(experiments.values())
        model_best_scores[model_type] = best_score
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = list(model_best_scores.keys())
    scores = list(model_best_scores.values())
    
    bars = ax.bar(models, scores)
    ax.set_title('Best Performance Comparison Across Models')
    ax.set_ylabel('Best F1-Score')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../plots/model_comparison.jpg', dpi=300, bbox_inches='tight', format='jpg')
    plt.show()
    
    # Print comparison
    print("\nBest F1-Scores by Model Type:")
    print("-" * 35)
    for model, score in model_best_scores.items():
        print(f"{model}: {score:.4f}")
    
    best_overall = max(model_best_scores.items(), key=lambda x: x[1])
    print(f"\nOverall best performing model: {best_overall[0]} (F1-Score: {best_overall[1]:.4f})")

def generate_report(results):
    """Generate a comprehensive report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE EXPERIMENT REPORT")
    print("="*60)
    
    # Create summary table
    summary_data = []
    for model_type, experiments in results.items():
        for exp_name, score in experiments.items():
            summary_data.append({
                'Model': model_type,
                'Experiment': exp_name,
                'F1_Score': score
            })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save to CSV
    df_summary.to_csv('../results/experiment_summary.csv', index=False)
    print("Summary table saved to results/experiment_summary.csv")
    
    # Statistical analysis
    print("\nSTATISTICAL SUMMARY:")
    print("-" * 30)
    
    for model_type in df_summary['Model'].unique():
        model_data = df_summary[df_summary['Model'] == model_type]['F1_Score']
        print(f"\n{model_type}:")
        print(f"  Mean F1-Score: {model_data.mean():.4f}")
        print(f"  Std F1-Score: {model_data.std():.4f}")
        print(f"  Min F1-Score: {model_data.min():.4f}")
        print(f"  Max F1-Score: {model_data.max():.4f}")

def main():
    """Main analysis function"""
    # Create plots directory
    os.makedirs('../plots', exist_ok=True)
    
    # Load results
    results = load_results()
    if results is None:
        return
    
    # Analyze each model type
    analyze_cnn_results(results)
    analyze_rnn_results(results)
    analyze_lstm_results(results)
    
    # Overall comparison
    compare_all_models(results)
    
    # Generate comprehensive report
    generate_report(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED!")
    print("="*60)
    print("\nGenerated files:")
    print("- plots/cnn_detailed_analysis.jpg")
    print("- plots/rnn_detailed_analysis.jpg")
    print("- plots/lstm_detailed_analysis.jpg")
    print("- plots/model_comparison.jpg")
    print("- results/experiment_summary.csv")

if __name__ == "__main__":
    main()
