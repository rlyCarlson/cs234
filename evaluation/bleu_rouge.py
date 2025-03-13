import numpy as np
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction 
from rouge_score import rouge_scorer

dfs = [pd.read_csv('/Users/ishaansingh/cs234/dev_fintuned_dpo_dummy_size.csv'), pd.read_csv('/Users/ishaansingh/cs234/dev_fintuned_dpo_dummy_v3.csv'), pd.read_csv('/Users/ishaansingh/cs234/dev_fintuned_dpo_dummy.csv')]
# BLEU
for df in dfs:
    df["gold_output"] = df["gold_output"].apply(lambda x: x.lower().split())
    df["model_output"] = df["model_output"].apply(lambda x: x.lower().split())

    smoother = SmoothingFunction().method1  
    df["BLEU"] = df.apply(lambda row: sentence_bleu(
        [row["gold_output"]], row["model_output"], smoothing_function=smoother), axis=1)

    average_bleu = df["BLEU"].mean()  
    print(f"Mean BLEU Score: {average_bleu:.4f}")

    def compute_rouge(reference, generated):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

    df[["ROUGE-1", "ROUGE-2", "ROUGE-L"]] = df.apply(
        lambda row: compute_rouge(" ".join(row["gold_output"]), " ".join(row["model_output"])), axis=1, result_type="expand"
    )

    average_rouge1 = df["ROUGE-1"].mean()
    average_rouge2 = df["ROUGE-2"].mean()
    average_rougeL = df["ROUGE-L"].mean()

    print(f"Mean ROUGE-1 Score: {average_rouge1:.4f}")
    print(f"Mean ROUGE-2 Score: {average_rouge2:.4f}")
    print(f"Mean ROUGE-L Score: {average_rougeL:.4f}")