import numpy as np
import pandas as pd

df_base = pd.read_csv('/Users/ishaansingh/cs234/dev_cpo_laj_results.csv')
df_fine = pd.read_csv('/Users/ishaansingh/cs234/dev_dpo_laj_results.csv')

average_values_base = df_base[['semantic_accuracy_score', 'stylistic_appropriateness_score', 'overall_score']].mean(axis=0)
average_values_fine = df_fine[['semantic_accuracy_score', 'stylistic_appropriateness_score', 'overall_score']].mean(axis=0)
print(average_values_base)
print(average_values_fine)