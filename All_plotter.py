# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:33:11 2025

@author: isotta
"""

import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


base_dir = os.getcwd()
All_dir = os.path.join(base_dir, '004')

arrays_other = []
arrays_stop = []
arrays_walk = []
arrays_jog = []
arrays_sprint = []
arrays_trans = []

names_other = []
names_stop = []
names_walk = []
names_jog = []
names_sprint = []
names_trans = []


# Find the positions of all the csv files inside the selected folder
csv_files = []
for file_name in os.listdir(All_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(All_dir, file_name)
        csv_files.append(file_path)

# Read every file in the folder and process the data
for file_path in csv_files:
    try:
        # Read the CSV file and transform it into a data frame (df)
        file_name = os.path.basename(file_path)
        df = pd.read_csv(file_path, header=None)
        df = df.values
        
        l=len(df[:,0])
        signal = np.zeros((l,5))
        act = df[0,3]
        signal[:,0:3] = df[:,0:3]
        signal[0,4]=df[1,3]
        
        if act == 0:
            signal[:,3] = np.ones(l)*(-1)
            arrays_other.append(signal)
            names_other.append(file_name)
        elif act == 1:
            signal[:,3] = np.zeros(l)
            arrays_stop.append(signal)
            names_stop.append(file_name)
        elif act == 2:
            signal[:,3] = np.ones(l)
            arrays_walk.append(signal)
            names_walk.append(file_name)
        elif act == 3:
            signal[:,3] = np.ones(l)*2
            arrays_jog.append(signal)
            names_jog.append(file_name)
        elif act == 4:
            signal[:,3] = np.ones(l)*3
            arrays_sprint.append(signal)
            names_sprint.append(file_name)
        elif act == 5:
            signal[:,3] = np.ones(l)*4
            arrays_trans.append(signal)
            names_trans.append(file_name)
        
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        
arrays = [arrays_other, arrays_stop, arrays_walk, arrays_jog, arrays_sprint, arrays_trans]
names = [names_other, names_stop, names_walk, names_jog, names_sprint, names_trans]
# Number of the activity: 0 other, 1 stop, 2 walk, 3 jog, 4 sprint, 5 tansition
a = 4
# Vertical limit of the plot
b = 15

pos_AP = [0, 1, 2, 9, 10, 11, 18, 19, 20]
pos_CC = [3, 4, 5, 12, 13, 14, 21, 22, 23]
pos_ML = [6, 7, 8, 15, 16, 17, 24, 25, 26]
pages = math.ceil(len(arrays[a]) / 9)
# Create the two pages (18 plots in total)
for page in range(pages):
    fig, axs = plt.subplots(9, 3, figsize=(18, 14))  # 3x3 grid of subplots per page
    fig.suptitle(f'Page {page+1}', fontsize=16)
    
    # Loop through each subplot on the page
    for i in range(9):
          # Get the appropriate subplot location (3x3 grid)
        
        # Get the index for the current plot (page * 27 + i)
        plot_index = page * 9 + i
        if plot_index >= len(arrays[a]):
            break
        
        plot_index_AP = page * 27 + pos_AP[i]
        plot_index_CC = page * 27 + pos_CC[i]
        plot_index_ML = page * 27 + pos_ML[i]
        
        ax_AP = axs[pos_AP[i] // 3, pos_AP[i] % 3]
        ax_CC = axs[pos_CC[i] // 3, pos_CC[i] % 3]
        ax_ML = axs[pos_ML[i] // 3, pos_ML[i] % 3]
        
        if a in [2,3,4]:
            steps = int(arrays[a][plot_index][0,4])
            ax_AP.set_title(names[a][plot_index] + f', steps = {steps}')
        else:
            ax_AP.set_title(names[a][plot_index])
        ax_AP.plot(arrays[a][plot_index][:, 0], color = 'black', label='Column 1')
        ax_AP.set_ylim(-b, b)
        
        ax_CC.plot(arrays[a][plot_index][:, 1], color ='green', label='Column 2')
        ax_CC.set_ylim(-b, b)
        
        ax_ML.plot(arrays[a][plot_index][:, 2], color = 'blue' ,label='Column 3')
        ax_ML.set_ylim(-b, b)
        
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust space for suptitle
    plt.show()