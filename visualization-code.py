import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    main()

def load_results(filename='medical_notes_simplification_results.json'):
    """Load results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found. Please run the main script first.")
        return []
        
def main():
    """Main function to run the analysis"""
    # Load the results
    results = load_results()
    
    if not results:
        print("No results found. Please run the medical_notes_simplification.py script first.")
        return
        
    # Plot readability comparisons
    print("Generating readability comparison plots...")
    plot_readability_comparisons(results)
    
    # Generate summary metrics
    print("\nGenerating summary metrics...\n")
    summary = compare_methods_summary(results)
    
    # Calculate additional statistics
    orig_readability = np.mean([r['original_complexity']['flesch_kincaid'] for r in results])
    best_method = summary.loc[summary['Readability Improvement'].idxmax(), 'Method']
    best_improvement = summary['Readability Improvement'].max()
    
    print("\nAdditional Statistics:")
    print(f"Original average readability: {orig_readability:.2f}")
    print(f"Best method: {best_method} (improvement: {best_improvement:.2f})")
    print(f"Medical terminology reduction with best method: {summary.loc[summary['Method'] == best_method, 'Medical Term Reduction'].values[0]:.2f}%")
    
    # Display a sample comparison from the first result
    if len(results) > 0:
        print("\nSample comparison from first patient:")
        print(f"Patient ID: {results[0]['patient_id']} (Education level: {results[0]['education_level']})")
        print("\nOriginal note:")
        print(results[0]['original_note'])
        print(f"\nBest method result ({best_method}):")
        if best_method == 'Basic':
            print(results[0]['basic_prompt_result'])
        elif best_method == 'In-Context':
            print(results[0]['in_context_result'])
        elif best_method == 'Chain-of-Thought':
            print(results[0]['chain_of_thought_result'])
        elif best_method == 'Tree-of-Thought':
            print(results[0]['tree_of_thought_result'])
        elif best_method == 'Combined':
            print(results[0]['combined_result'])
    
    print("\nAnalysis complete!")

def plot_readability_comparisons(results):
    """Create visualizations comparing readability metrics across different methods."""
    plt.figure(figsize=(14, 8))
    
    # Extract Flesch-Kincaid scores for all methods
    fk_scores = {
        'Original': [r['original_complexity']['flesch_kincaid'] for r in results],
        'Basic': [r['basic_complexity']['flesch_kincaid'] for r in results],
        'In-Context': [r['in_context_complexity']['flesch_kincaid'] for r in results],
        'Chain-of-Thought': [r['cot_complexity']['flesch_kincaid'] for r in results],
        'Tree-of-Thought': [r['tot_complexity']['flesch_kincaid'] for r in results],
        'Combined': [r['combined_complexity']['flesch_kincaid'] for r in results]
    }
    
    # Prepare data for plotting
    methods = list(fk_scores.keys())
    x = np.arange(len(results))
    width = 0.12
    offsets = np.arange(-2.5, 3.5) * width
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        plt.bar(x + offsets[i], fk_scores[method], width, label=method)
    
    plt.xlabel('Patient')
    plt.ylabel('Flesch-Kincaid Readability (higher is better)')
    plt.title('Readability Comparison Across Methods')
    plt.xticks(x, [r['patient_id'] for r in results])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal lines for readability guidelines
    plt.axhline(y=80, color='g', linestyle='-', alpha=0.3)
    plt.axhline(y=60, color='y', linestyle='-', alpha=0.3)
    plt.axhline(y=40, color='r', linestyle='-', alpha=0.3)
    
    plt.text(len(results)-1, 85, 'Easy', color='g', ha='right')
    plt.text(len(results)-1, 65, 'Standard', color='y', ha='right')
    plt.text(len(results)-1, 45, 'Difficult', color='r', ha='right')
    
    plt.tight_layout()
    plt.savefig('readability_comparison.png')
    plt.show()
    
    # Plot average sentence length comparison
    plt.figure(figsize=(14, 6))
    
    # Extract average sentence lengths
    sent_lengths = {
        'Original': [r['original_complexity']['avg_sentence_length'] for r in results],
        'Basic': [r['basic_complexity']['avg_sentence_length'] for r in results],
        'In-Context': [r['in_context_complexity']['avg_sentence_length'] for r in results],
        'Chain-of-Thought': [r['cot_complexity']['avg_sentence_length'] for r in results],
        'Tree-of-Thought': [r['tot_complexity']['avg_sentence_length'] for r in results],
        'Combined': [r['combined_complexity']['avg_sentence_length'] for r in results]
    }
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        plt.bar(x + offsets[i], sent_lengths[method], width, label=method)
    
    plt.xlabel('Patient')
    plt.ylabel('Average Sentence Length (words)')
    plt.title('Sentence Length Comparison Across Methods')
    plt.xticks(x, [r['patient_id'] for r in results])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add reference line for recommended sentence length
    plt.axhline(y=15, color='r', linestyle='--', alpha=0.5)
    plt.text(len(results)-1, 16, 'Recommended max. for general audience', color='r', ha='right')
    
    plt.tight_layout()
    plt.savefig('sentence_length_comparison.png')
    plt.show()
    
    # Compare medical terms percentage
    plt.figure(figsize=(14, 6))
    
    # Extract medical terms percentages
    med_terms = {
        'Original': [r['original_complexity']['medical_terms_pct'] for r in results],
        'Basic': [r['basic_complexity']['medical_terms_pct'] for r in results],
        'In-Context': [r['in_context_complexity']['medical_terms_pct'] for r in results],
        'Chain-of-Thought': [r['cot_complexity']['medical_terms_pct'] for r in results],
        'Tree-of-Thought': [r['tot_complexity']['medical_terms_pct'] for r in results],
        'Combined': [r['combined_complexity']['medical_terms_pct'] for r in results]
    }
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        plt.bar(x + offsets[i], med_terms[method], width, label=method)
    
    plt.xlabel('Patient')
    plt.ylabel('Medical Terms (%)')
    plt.title('Medical Terminology Usage Across Methods')
    plt.xticks(x, [r['patient_id'] for r in results])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('medical_terms_comparison.png')
    plt.show()

def compare_methods_summary(results):
    """Create a summary comparison of all methods"""
    # Calculate average metrics across all patients
    avg_readability = {
        'Original': np.mean([r['original_complexity']['flesch_kincaid'] for r in results]),
        'Basic': np.mean([r['basic_complexity']['flesch_kincaid'] for r in results]),
        'In-Context': np.mean([r['in_context_complexity']['flesch_kincaid'] for r in results]),
        'Chain-of-Thought': np.mean([r['cot_complexity']['flesch_kincaid'] for r in results]),
        'Tree-of-Thought': np.mean([r['tot_complexity']['flesch_kincaid'] for r in results]),
        'Combined': np.mean([r['combined_complexity']['flesch_kincaid'] for r in results])
    }
    
    avg_med_terms = {
        'Original': np.mean([r['original_complexity']['medical_terms_pct'] for r in results]),
        'Basic': np.mean([r['basic_complexity']['medical_terms_pct'] for r in results]),
        'In-Context': np.mean([r['in_context_complexity']['medical_terms_pct'] for r in results]),
        'Chain-of-Thought': np.mean([r['cot_complexity']['medical_terms_pct'] for r in results]),
        'Tree-of-Thought': np.mean([r['tot_complexity']['medical_terms_pct'] for r in results]),
        'Combined': np.mean([r['combined_complexity']['medical_terms_pct'] for r in results])
    }
    
    avg_sentence_length = {
        'Original': np.mean([r['original_complexity']['avg_sentence_length'] for r in results]),
        'Basic': np.mean([r['basic_complexity']['avg_sentence_length'] for r in results]),
        'In-Context': np.mean([r['in_context_complexity']['avg_sentence_length'] for r in results]),
        'Chain-of-Thought': np.mean([r['cot_complexity']['avg_sentence_length'] for r in results]),
        'Tree-of-Thought': np.mean([r['tot_complexity']['avg_sentence_length'] for r in results]),
        'Combined': np.mean([r['combined_complexity']['avg_sentence_length'] for r in results])
    }
    
    # Plot comparison of average metrics
    plt.figure(figsize=(14, 8))
    
    methods = list(avg_readability.keys())
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot readability scores
    ax.bar(x - width, list(avg_readability.values()), width, label='Readability Score (higher is better)')
    
    # Plot medical terms percentage on the same scale
    scaled_med_terms = [value * 5 for value in avg_med_terms.values()]  # Scale up for visibility
    ax.bar(x, scaled_med_terms, width, label='Medical Terms % (×5)')
    
    # Plot sentence length on the same scale
    scaled_sentence_length = [value * 3 for value in avg_sentence_length.values()]  # Scale up for visibility
    ax.bar(x + width, scaled_sentence_length, width, label='Avg Sentence Length (×3)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylabel('Score')
    ax.set_title('Average Metrics Across All Prompting Methods')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('average_metrics_comparison.png')
    plt.show()
    
    # Create a summary table
    summary = pd.DataFrame({
        'Method': methods,
        'Avg. Readability': list(avg_readability.values()),
        'Avg. Medical Term %': list(avg_med_terms.values()),
        'Avg. Sentence Length': list(avg_sentence_length.values()),
        'Readability Improvement': [val - avg_readability['Original'] for val in avg_readability.values()],
        'Medical Term Reduction': [avg_med_terms['Original'] - val for val in avg_med_terms.values()]
    })
    
    print("Summary of Method Performance:")
    print(summary)
    
    return summary