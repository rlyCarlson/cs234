#!/usr/bin/env python3
"""
LLM Judge Interface

A script that uses LLMs to judge language model outputs against gold standards,
providing qualitative evaluation alongside automatic metrics.
"""

import argparse
import pandas as pd
import numpy as np
import json
import time
import os
import csv
from tqdm import tqdm
import anthropic
from typing import List, Dict, Any, Optional, Union, Tuple

class LLMJudge:
    """Base class for LLM judges."""
    
    def evaluate_example(self, instruction: str, input_text: str, model_output: str, gold_output: str) -> Dict:
        """Evaluate a single example using the LLM."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def evaluate_dataset(self, dataset_path: str, output_path: str, sample_size: Optional[int] = None) -> None:
        """Evaluate a dataset of examples."""
        # Load dataset
        df = pd.read_csv(dataset_path)
        required_cols = ['model_output', 'gold_output']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns. Required: {required_cols}")
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
        
        results = []
        
        # Process each example
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating with LLM"):
            instruction = row.get('instruction', 'Perform the task correctly')
            input_text = row.get('input', '')
            model_output = row['model_output']
            gold_output = row['gold_output']
            
            # Skip empty outputs
            if pd.isna(model_output) or pd.isna(gold_output):
                continue
                
            # Get evaluation from LLM
            evaluation = self.evaluate_example(instruction, input_text, model_output, gold_output)
            
            # Add to results
            result = {
                'index': idx,
                'instruction': instruction,
                'input': input_text,
                'model_output': model_output,
                'gold_output': gold_output
            }
            
            # Add all evaluation fields
            for k, v in self._flatten_dict(evaluation):
                result[k] = v
            
            results.append(result)
            
            # Save after each evaluation to avoid losing progress
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_path, index=False)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Save summary
        summary_path = output_path.replace('.csv', '_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Score'])
            for metric, score in summary.items():
                writer.writerow([metric, score])
        
        print(f"\n===== SUMMARY =====")
        for metric, score in summary.items():
            print(f"{metric}: {score}")
        print(f"Detailed results saved to {output_path}")
        print(f"Summary saved to {summary_path}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '') -> List[Tuple[str, Any]]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items
    
    def _calculate_summary(self, results: List[Dict]) -> Dict[str, Union[float, int, str]]:
        """Calculate summary statistics from results."""
        # Override in subclasses as needed
        return {"count": len(results)}


class ClaudeJudge(LLMJudge):
    """LLM Judge using Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """Initialize the Claude API client."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set ANTHROPIC_API_KEY environment variable or pass it as an argument.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
    
    def create_prompt(self, instruction: str, input_text: str, model_output: str, gold_output: str) -> str:
        """Create a prompt for Claude to evaluate the model output."""
        prompt = f"""
You are a helpful and fair judge evaluating the performance of language models on a specific task.

Task: {instruction}
Input: {input_text}

Model Output: {model_output}

Gold Standard Output: {gold_output}

Please evaluate the model output compared to the gold standard on the following criteria:
1. Semantic Accuracy: How well does the model capture the core meaning and intent of the gold standard? (Score 1-10)
2. Stylistic Appropriateness: How well does the model match the required style, tone, and format for the task? (Score 1-10)

For each criterion, provide:
- A numerical score (1-10)
- A brief explanation for the score

Then give an overall score (1-10) and a concise verdict on the model's performance for this example.

Format your response as a JSON object with the following structure:
{{
  "semantic_accuracy": {{
    "score": <score>,
    "explanation": "<explanation>"
  }},
  "stylistic_appropriateness": {{
    "score": <score>,
    "explanation": "<explanation>"
  }},
  "overall": {{
    "score": <score>,
    "verdict": "<verdict>"
  }}
}}
"""
        return prompt

    def evaluate_example(self, instruction: str, input_text: str, model_output: str, gold_output: str, max_retries: int = 3) -> Dict:
        """Evaluate a single example using Claude."""
        prompt = self.create_prompt(instruction, input_text, model_output, gold_output)
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.0,  # Using 0 for deterministic evaluation
                    system="You are a fair and consistent judge for evaluating language model outputs. Always provide your evaluation in valid JSON format.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract JSON from response
                try:
                    content = response.content[0].text
                    # Find JSON in the content
                    json_str = content
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        json_str = content.split("```")[1].split("```")[0].strip()
                    
                    evaluation = json.loads(json_str)
                    return evaluation
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"Error parsing JSON from Claude response: {e}")
                    print(f"Raw response: {content}")
                    if attempt == max_retries - 1:
                        return {
                            "accuracy": {"score": 0, "explanation": "Failed to parse evaluation"},
                            "fluency": {"score": 0, "explanation": "Failed to parse evaluation"},
                            "task_fulfillment": {"score": 0, "explanation": "Failed to parse evaluation"},
                            "overall": {"score": 0, "verdict": "Failed to parse evaluation"}
                        }
            except anthropic.RateLimitError:
                print(f"Rate limit exceeded. Waiting before retry {attempt+1}/{max_retries}...")
                time.sleep(20 * (attempt + 1))  # Exponential backoff
            except Exception as e:
                print(f"Error calling Claude API: {e}")
                if attempt == max_retries - 1:
                    return {
                        "accuracy": {"score": 0, "explanation": f"API error: {str(e)}"},
                        "fluency": {"score": 0, "explanation": f"API error: {str(e)}"},
                        "task_fulfillment": {"score": 0, "explanation": f"API error: {str(e)}"},
                        "overall": {"score": 0, "verdict": f"API error: {str(e)}"}
                    }
                time.sleep(5)
        
        # If we reach here, all retries failed
        return {
            "accuracy": {"score": 0, "explanation": "Max retries exceeded"},
            "fluency": {"score": 0, "explanation": "Max retries exceeded"},
            "task_fulfillment": {"score": 0, "explanation": "Max retries exceeded"},
            "overall": {"score": 0, "verdict": "Max retries exceeded"}
        }
    
    def _calculate_summary(self, results: List[Dict]) -> Dict[str, Union[float, int, str]]:
        """Calculate summary statistics from Claude evaluation results."""
        summary = {
            'count': len(results),
            'avg_accuracy_score': f"{np.mean([r['accuracy_score'] for r in results]):.2f}/10",
            'avg_fluency_score': f"{np.mean([r['fluency_score'] for r in results]):.2f}/10",
            'avg_task_fulfillment_score': f"{np.mean([r['task_fulfillment_score'] for r in results]):.2f}/10",
            'avg_overall_score': f"{np.mean([r['overall_score'] for r in results]):.2f}/10"
        }
        return summary


def main():
    parser = argparse.ArgumentParser(description='Use LLMs to evaluate model outputs against gold standards.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='llm_evaluation_results.csv', help='Path to save the results CSV')
    parser.add_argument('--api_key', type=str, help='Anthropic API key (optional if set as environment variable)')
    parser.add_argument('--model', type=str, default='claude-3-sonnet-20240229', help='Claude model to use')
    parser.add_argument('--sample', type=int, help='Number of examples to sample for evaluation (optional)')
    args = parser.parse_args()
    
    try:
        judge = ClaudeJudge(
            api_key=args.api_key,
            model=args.model
        )
        print(f"Using Claude model: {args.model}")
        
        # Run evaluation
        judge.evaluate_dataset(
            dataset_path=args.input, 
            output_path=args.output,
            sample_size=args.sample
        )
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()