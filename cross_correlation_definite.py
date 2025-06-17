import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.signal import correlate
from scipy.stats import mode

def extract_real_activity_windows(real_activities, window_size=24, overlap=12):
    window_labels = []
    for start in range(0, len(real_activities) - window_size, overlap):
        window = real_activities[start:start + window_size]
        most_common = mode(window, keepdims=False).mode
        window_labels.append(most_common)
    return window_labels
def compress_signal(signal, window_size=24, overlap=12):
    compressed = []
    for start in range(0, len(signal) - window_size, overlap):
        window = signal[start:start + window_size]
        compressed.append(np.mean(window[:, 0]))  # o[:, i] per altri assi
    return compressed

# 📌 Function to select a folder
def select_folder(title="Select a folder"):
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title=title)
    return folder_selected

# 📌 Function to load signals from CSV files
def load_signals_from_folder():
    folder = select_folder("Select the folder containing the activity CSV files")
    if not folder:
        print("❌ No folder selected. Operation cancelled.")
        return [], []

    signals, activities = [], []
    data_list = []
    
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, file), header=None)
            signal = df.iloc[:, :3].values
            activity = int(df.iloc[0, 3])
            
            data_list.append((activity, signal))
    
    # Sort data by activity
    data_list.sort(key=lambda x: x[0])
    
    for activity, signal in data_list:
        signals.append(signal)
        activities.extend([activity] * len(signal))
    
    return np.concatenate(signals, axis=0), np.array(activities)

# 📌 Function to compute sliding window correlation
def compute_sliding_correlation(signal, ref_signal, window_size=24, overlap=12):
    num_samples = signal.shape[0]
    correlation_values = []
    
    for start in range(0, num_samples - window_size, overlap):
        window = signal[start:start + window_size, :]
        correlation = [correlate(window[:, i], ref_signal[:, i], mode='valid').max() for i in range(3)]
        avg_correlation = np.mean(correlation)
        correlation_values.append(avg_correlation)
    
    # Normalize between -1 and 1
    correlation_values = np.array(correlation_values)
    if len(correlation_values) > 0:
        correlation_values = 2 * (correlation_values - np.min(correlation_values)) / (np.max(correlation_values) - np.min(correlation_values) + 1e-6) - 1
    
    return correlation_values

# 📌 Load reference steps
def load_reference_steps():
    folder = select_folder("Select the folder containing the athlete's reference steps")
    if not folder:
        return {}

    ref_steps = {}
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, file), header=None)
            activity = int(df.iloc[0, 3])
            ref_signal = df.iloc[:, :3].values
            ref_signal = (ref_signal - np.mean(ref_signal, axis=0)) / (np.std(ref_signal, axis=0) + 1e-6)  # Normalization
            ref_steps[activity] = ref_signal
    return ref_steps

# 📌 Load data
dataset_signal, real_activities = load_signals_from_folder()
reference_steps = load_reference_steps()

# 📌 Compute sliding correlations
corr_results = {}
for activity in [2, 3, 4]:
    if activity in reference_steps:
        corr_results[activity] = compute_sliding_correlation(dataset_signal, reference_steps[activity])
    else:
        corr_results[activity] = np.zeros(len(real_activities) // 12)  # If no data available, fill with zeros

# 📌 Predict activity based on highest correlation
predicted_activities = []
window_count = len(next(iter(corr_results.values())))
real_activities_windows = extract_real_activity_windows(real_activities)

predicted_activities = []
window_size = 24
overlap = 12

for i in range(0, len(dataset_signal) - window_size, overlap):
    window = dataset_signal[i:i + window_size]
    # Check if all acceleration values are within [-1, 1] → activity 1 (Stop)
    # if np.all(np.std(window, axis=0) < 0.2):
    #    predicted_activities.append(1)
    #    continue


    # Otherwise, use correlation
    max_corr = -np.inf
    best_activity = 1  # default to Stop if nothing matches
    window_index = i // overlap

    for activity in [2, 3, 4]:
        if i < len(corr_results[activity]) and corr_results[activity][i] > max_corr: #and corr_results[activity][i] > -0.8:
            max_corr = corr_results[activity][i]
            best_activity = activity
    
    predicted_activities.append(best_activity)

# 📌 Plotting
plt.figure(figsize=(12, 10))

# 1. Acceleration signal
plt.subplot(5, 1, 1)
plt.plot(dataset_signal [:,0], label="Signal X ")
plt.title("Acceleration Signal")
plt.xlabel("Samples")
plt.ylabel("Acceleration")
plt.legend()

# 2-4. Sliding correlation with Walk, Jog, and Sprint
activities_labels = {2: "Walk", 3: "Jog", 4: "Sprint"}
for i, activity in enumerate([2, 3, 4]):
    plt.subplot(5, 1, i + 2)
    plt.plot(corr_results[activity], label=f"Similarity with {activities_labels[activity]}", color='purple')
    plt.ylim(-1, 1)
    plt.title(f"Similarity with {activities_labels[activity]} (Normalized -1 to 1)")
    plt.xlabel("Window")
    plt.ylabel("Correlation")
    plt.legend()

# 5. Real vs Predicted activity
plt.subplot(5, 1, 5)
plt.plot(real_activities_windows, label="Real Activity", color='green')
plt.plot(predicted_activities, label="Predicted Activity", color='blue', linestyle='dashed')
plt.title("Real vs Predicted Activity Comparison")
plt.xlabel("Windows")
plt.ylabel("Activity")
plt.legend()

plt.tight_layout()
plt.show()
