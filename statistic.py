# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:38:16 2025

@author: isotta
"""

import pandas as pd
import os
from sklearn.metrics import classification_report

# Path with Excel files
input_folder = r'C:\Users\isotta\Desktop\dati Inail\intervalli\features\L1O_features\overall matrix'
output_file = 'metrics_for_each_class.xlsx'

# Lists for data collection
all_true = []
all_pred = []

# Get trough all the files on the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        file_path = os.path.join(input_folder, filename)
        print(f"Sto processando: {filename}")

        # read file Excel (1st and 2nd columns)
        df = pd.read_excel(file_path, usecols=[0, 1], names=['true_label', 'eval_label'])

        # Rimuovi righe vuote
        df = df.dropna()

        # Tieni solo le righe con label validi (1, 2, 3, 4)
        df = df[df['true_label'].isin([1, 2, 3, 4])]
        df = df[df['eval_label'].isin([1, 2, 3, 4])]

        # Aggiungi ai dati complessivi
        all_true.extend(df['true_label'].tolist())
        all_pred.extend(df['eval_label'].tolist())

# Macro report for each activity class
report = classification_report(all_true, all_pred, labels=[1, 2, 3, 4],
                               target_names=['stop', 'walk', 'jog', 'sprint'], output_dict=True, zero_division=0)

# To DataFrame for saving
report_df = pd.DataFrame(report).transpose()

# Saving in Excel
report_df.to_excel(output_file)

print(f"Metrics saved in : {output_file}")
