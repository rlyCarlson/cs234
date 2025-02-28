#!/usr/bin/env python3
"""
Evaluation script for comparing LLM outputs with gold standards using BLEU and ROUGE metrics.
"""

import pandas as pd
import numpy as np
import csv
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
from tqdm import tqdm

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    print("Warning: Unable to download NLTK data. If NLTK is not installed, please install it with 'pip install nltk'.")

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        required_cols = ['model_output', 'gold_output']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: CSV missing required columns. Required: {required_cols}")
            print(f"Found: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def tokenize(text):
    """Tokenize the text into words."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    return nltk.word_tokenize(text.lower())

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score between reference and hypothesis."""
    if not reference or not hypothesis:
        return 0.0
    
    # Tokenize
    reference_tokens = [tokenize(reference)]
    hypothesis_tokens = tokenize(hypothesis)
    
    # Check if tokens exist
    if not reference_tokens[0] or not hypothesis_tokens:
        return 0.0
    
    # Use smoothing function to avoid score of 0 for short sentences
    smoothie = SmoothingFunction().method1
    
    # Calculate BLEU for different n-gram weightings
    try:
        bleu1 = sentence_bleu(reference_tokens, hypothesis_tokens, 
                             weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu(reference_tokens, hypothesis_tokens, 
                             weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu(reference_tokens, hypothesis_tokens, 
                             weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        return {
            'bleu-1': bleu1,
            'bleu-2': bleu2,
            'bleu-4': bleu4
        }
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        print(f"Reference: {reference}")
        print(f"Hypothesis: {hypothesis}")
        return {
            'bleu-1': 0.0,
            'bleu-2': 0.0,
            'bleu-4': 0.0
        }

def calculate_rouge(reference, hypothesis):
    """Calculate ROUGE scores between reference and hypothesis."""
    if pd.isna(reference) or pd.isna(hypothesis) or not isinstance(reference, str) or not isinstance(hypothesis, str):
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)[0]
        return scores
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        print(f"Reference: {reference}")
        print(f"Hypothesis: {hypothesis}")
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}

def evaluate_dataset(df):
    """Evaluate the model outputs against gold standards."""
    results = []
    
    # Initialize aggregated scores
    all_bleu1 = []
    all_bleu2 = []
    all_bleu4 = []
    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        model_output = row['model_output']
        gold_output = row['gold_output']
        
        # Calculate BLEU and ROUGE scores
        bleu_scores = calculate_bleu(gold_output, model_output)
        rouge_scores = calculate_rouge(gold_output, model_output)
        
        # Add scores to aggregated lists
        all_bleu1.append(bleu_scores['bleu-1'])
        all_bleu2.append(bleu_scores['bleu-2'])
        all_bleu4.append(bleu_scores['bleu-4'])
        all_rouge1.append(rouge_scores['rouge-1']['f'])
        all_rouge2.append(rouge_scores['rouge-2']['f'])
        all_rougeL.append(rouge_scores['rouge-l']['f'])
        
        # Save individual results
        result = {
            'index': idx,
            'bleu-1': bleu_scores['bleu-1'],
            'bleu-2': bleu_scores['bleu-2'],
            'bleu-4': bleu_scores['bleu-4'],
            'rouge-1-f': rouge_scores['rouge-1']['f'],
            'rouge-2-f': rouge_scores['rouge-2']['f'],
            'rouge-l-f': rouge_scores['rouge-l']['f'],
            'model_output': model_output,
            'gold_output': gold_output
        }
        
        # Add instruction and input if available
        if 'instruction' in df.columns:
            result['instruction'] = row['instruction']
        if 'input' in df.columns:
            result['input'] = row['input']
        
        results.append(result)
    
    # Calculate average scores
    avg_scores = {
        'avg_bleu-1': np.mean(all_bleu1),
        'avg_bleu-2': np.mean(all_bleu2),
        'avg_bleu-4': np.mean(all_bleu4),
        'avg_rouge-1-f': np.mean(all_rouge1),
        'avg_rouge-2-f': np.mean(all_rouge2),
        'avg_rouge-l-f': np.mean(all_rougeL),
    }
    
    return results, avg_scores

def save_results(results, avg_scores, output_file):
    """Save evaluation results to a CSV file."""
    try:
        # Save detailed results
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"Detailed results saved to {output_file}")
        
        # Print summary
        print("\n===== SUMMARY METRICS =====")
        print(f"Average BLEU-1: {avg_scores['avg_bleu-1']:.4f}")
        print(f"Average BLEU-2: {avg_scores['avg_bleu-2']:.4f}")
        print(f"Average BLEU-4: {avg_scores['avg_bleu-4']:.4f}")
        print(f"Average ROUGE-1-F: {avg_scores['avg_rouge-1-f']:.4f}")
        print(f"Average ROUGE-2-F: {avg_scores['avg_rouge-2-f']:.4f}")
        print(f"Average ROUGE-L-F: {avg_scores['avg_rouge-l-f']:.4f}")
        
        # Save summary to a separate file
        summary_file = output_file.replace('.csv', '_summary.csv')
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Score'])
            for metric, score in avg_scores.items():
                writer.writerow([metric, f"{score:.4f}"])
        print(f"Summary metrics saved to {summary_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Calculate BLEU and ROUGE scores between model outputs and gold standards.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='evaluation_results.csv', help='Path to save the results CSV')
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    
    if df is not None and not df.empty:
        print(f"Loaded {len(df)} examples. Starting evaluation...")
        results, avg_scores = evaluate_dataset(df)
        save_results(results, avg_scores, args.output)
        print("Evaluation completed successfully!")
    else:
        print("No data to evaluate. Please check your input file.")

if __name__ == "__main__":
    main()