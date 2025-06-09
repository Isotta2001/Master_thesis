# -*- coding: utf-8 -*-
"""
Created on Tue May  6 10:24:21 2025

@author: isotta
"""

import os
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, find_peaks
from decimal import Decimal, getcontext, ROUND_HALF_UP

#Funzione che trova le directory
def loader(name):
    base_dir = os.getcwd()
    directory = os.path.join(base_dir, name)
    test = os.path.join(directory, 'Test')
    train = os.path.join(directory, 'Train')
    return base_dir, directory, test, train

#Funzione che trova tutti i file csv in una cartella
def csv_finder (directory):
    csv_files = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            csv_files.append(file_path)
    
    files = []
    for file_path in csv_files:
        try:
            file_name = os.path.basename(file_path)
            df = pd.read_csv(file_path, header=None)
            df = df.values
            
            files.append(df)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            
    return files

# Funzione che elabora le proprietà nelle frequenze
def wave_eval (AP, CC, ML, y):
    step = 48
    step_len = 24

    AP_h = []
    CC_max = []
    label = []
    
    for j in range(0,len(AP)-step_len,step_len):
        signal_AP = AP[j:j+step]
        w_AP, _ = waveletter(signal_AP)
        
        signal_CC = CC[j:j+step]
        w_CC, _ = waveletter(signal_CC)
        
        signal_ML = ML[j:j+step]
        w_ML, _ = waveletter(signal_ML)
        
        _, h_AP = mid_high(w_AP)
        max_CC = np.max(np.abs(signal_CC))

        
        l = y[j]
        
        AP_h.append(h_AP)
        CC_max.append(max_CC)
        label.append(l)
        
    properties = [AP_h, CC_max, label]
    properties = np.vstack(properties).T
    return properties

#Funzione che trasforma il segnale nelle frequenze
def waveletter (data):
    S = 100
    scales = np.arange(1,S)
    wavelet = 'cmor1.0-0.5'

    coeffs, freqs = pywt.cwt(data, scales, wavelet)
    mean_coeffs = np.zeros(len(coeffs[:,0]))
    for j in range (len (mean_coeffs)):
        mean_coeffs[j] = np.mean(abs(coeffs[j,:]))
    x = mean_coeffs[::-1]       
    r = np.sqrt(np.mean(np.square(data)))
    return x, r

# Funzione che trova i picchi delle frequenze medie e alte
def mid_high (data):
    m = max(data[39:59])
    h = max(data[89:99])
    return m, h

#Funzione che riporta le whiskers e può plottare i boxplot
def boxplotter (arrays, names, title, flag, a = None, b = None):
    # Find the maximum length of the input arrays
    #arrays in the shape of [w,x,y,...]
    
    max_length = max(len(a) for a in arrays)
    
    # Pad arrays to have the same length (with NaN, for example)
    padded_arrays = [
        np.pad(a, (0, max_length - len(a)), mode='constant', constant_values=np.nan)
        for a in arrays
    ]
    
    array = np.array(padded_arrays).T
    df = pd.DataFrame(array, columns=names)
    
    lower_whiskers = []
    upper_whiskers = []

    # Calculate whiskers for each column
    for col in df.columns:
        q1 = np.nanpercentile(df[col], 25)  # 25th percentile (Q1)
        q3 = np.nanpercentile(df[col], 75)  # 75th percentile (Q3)
        iqr = q3 - q1  # Interquartile range
        
        # Calculate lower and upper whiskers (theoretical bounds)
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr
        
        # Clip whiskers to the min/max values within the bounds
        lower_whisker = max(df[col][df[col] >= lower_whisker].min(), lower_whisker)
        upper_whisker = min(df[col][df[col] <= upper_whisker].max(), upper_whisker)
        
        # Store whiskers in dictionary
        lower_whiskers.append(lower_whisker)
        upper_whiskers.append(upper_whisker)
    
    if flag == 1:
        plt.figure(figsize=(10, 6))  # Set the figure size
        sns.boxplot(data=df)
        plt.title(f'{title}')
        if a is not None and b is not None:
            plt.ylim(a,b)
        plt.show()
    
    return lower_whiskers, upper_whiskers

#Funzione che ritorna l'array dei label
def prop_eval (x,y,t1,t2,t3):
    l = []
    for i in range(len(x)):
        l.append(act_identifier(x[i],y[i],t1,t2,t3))
    l = np.array(l)
    return l

#Funzione che identifica l'attività
def act_identifier (h_AP, CC_max, t1, t2, t3):
    
    if CC_max < t1:
        act = 1
    elif h_AP < t2:
        act = 2
    elif h_AP < t3:
        act = 3
    else:
        act = 4

    return act

#Funzione che calcola la confusion matrix
def confusioner (y, l, flags=None):
    s_s = 0; s_w = 0; s_j = 0; s_r = 0
    w_s = 0; w_w = 0; w_j = 0; w_r = 0
    j_s = 0; j_w = 0; j_j = 0; j_r = 0
    r_s = 0; r_w = 0; r_j = 0; r_r = 0
    
    if flags is not None and np.any(flags):
        positions = [i for i, val in enumerate(flags) if val == 1]
        for pos in positions:
            y[pos] = None
            l[pos] = None

    for j in range(len(y)):
        if y[j] == 1:
            if l[j] == 1:
                s_s += 1
            elif l[j] == 2:
                s_w += 1
            elif l[j] == 3:
                s_j += 1
            elif l[j] == 4:
                s_r += 1
        elif y[j] == 2:
            if l[j] == 1:
                w_s += 1
            elif l[j] == 2:
                w_w += 1
            elif l[j] == 3:
                w_j += 1
            elif l[j] == 4:
                w_r += 1
        elif y[j] == 3:
            if l[j] == 1:
                j_s += 1
            elif l[j] == 2:
                j_w += 1
            elif l[j] == 3:
                j_j += 1
            elif l[j] == 4:
                j_r += 1
        elif y[j] == 4:
            if l[j] == 1:
                r_s += 1
            elif l[j] == 2:
                r_w += 1
            elif l[j] == 3:
                r_j += 1
            elif l[j] == 4:
                r_r += 1

    s = np.sum(y == 1)
    w = np.sum(y == 2)
    j = np.sum(y == 3)
    r = np.sum(y == 4)

    s_s /= s; s_w /= s; s_j /= s; s_r /= s
    w_s /= w; w_w /= w; w_j /= w; w_r /= w
    j_s /= j; j_w /= j; j_j /= j; j_r /= j
    r_s /= r; r_w /= r; r_j /= r; r_r /= r


    conf_matrx = np.array([
        [s_s, s_w, s_j, s_r],  
        [w_s, w_w, w_j, w_r],   
        [j_s, j_w, j_j, j_r],   
        [r_s, r_w, r_j, r_r]    
    ])
    
    return conf_matrx

def plot_conf(conf_mtrx,title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mtrx, annot=True, fmt='g', cmap='GnBu', xticklabels=['Stop', 'Walk', 'Jog', 'Sprint'], yticklabels=['Stop', 'Walk', 'Jog', 'Sprint'])

    # Adding labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()
    
def stepcounter_thresh(data,thresh,flag=0,flag2=0):
    signal = np.ravel(data)
    t = thresh


    minima ,_ = find_peaks(-signal)
    m = np.mean(signal[minima])
    maxima ,_ = find_peaks(signal)
    M = np.mean(signal[maxima])
    a = signal.shape
    signal = 2*((signal-np.ones(a)*m)/(M-m))-1


    signal[signal<0]=0
    pks,properties = find_peaks(signal, height = (np.mean(signal)*t))
    
    if flag ==1:
        plt.figure()
        plt.plot(signal)
        plt.plot(pks, signal[pks], 'ro', label="Peaks")
        plt.title('Walking segment')
        plt.show()
    
    if flag2 == 0:
        steps = len(pks)/2
    else:
        steps = len(pks)
    
    return steps

def thresh_calibrate(data, steps, flag=0):
    from decimal import Decimal, getcontext

    getcontext().prec = 6  # Set precision to what you need
    thresh = [float(Decimal('0.1') * i) for i in range(1, 51)]  # 0.1 to 5.0
    rel_err = []

    for t in thresh:
        if flag == 0:
            steps_iter = stepcounter_thresh(data, t)
        else:
            steps_iter = stepcounter_thresh(data, t, 0, 1)
        err = abs((steps_iter - steps) / steps) * 100
        rel_err.append(err)
    
    rel_err = np.array(rel_err)
    min_err = np.min(rel_err)
    min_index = np.argmin(rel_err)  # This gets the FIRST occurrence of the min
    best_thresh = round(thresh[min_index], 2)

    #print(f"rel err: {rel_err}")
    #print(f"best_thresh: {best_thresh}")

    return best_thresh
def round_to_step(x, step=0.05):
    d = Decimal(str(x))
    step = Decimal(str(step))
    return float((d / step).to_integral_value(rounding=ROUND_HALF_UP) * step)

def decimal_range(start, stop, step, precision=2):
    getcontext().prec = 6
    start = Decimal(str(start))
    stop = Decimal(str(stop))
    step = Decimal(str(step))
    result = []
    while start <= stop:
        result.append(float(start.quantize(Decimal(f"1.{'0'*precision}"))))
        start += step
    return result




#%% MAIN

base_dir, athl_dir , Test_dir, Train_dir = loader('005')
Test_files = csv_finder(Test_dir)
Train_files = csv_finder(Train_dir)
athlete_id = os.path.basename(athl_dir)

true_label_train = []
eval_label_train = []
true_label_test = []
eval_label_test = []

err_walk_train = 0
err_walk_test = 0
steps_walk_train = 0
steps_walk_test = 0

err_jog_train = 0
err_jog_test = 0
steps_jog_train = 0
steps_jog_test = 0

err_run_train = 0
err_run_test = 0
steps_run_train = 0
steps_run_test = 0

best_thresholds_list = []
stepcounting_thresholds_list = []

for i in range(len(Train_files)): #(len(Train_files)):
    
    print(f'Initiating cycle {i}')
    
    #%% Threshold evaluation
    test = Test_files[i]
    train = Train_files[i]
    
    #Estraction of acceleration data and labels
    AP = train[:,0]
    CC = train[:,1]
    ML = train[:,2]
    Y = train[:,3]
    
    #Finding ideal treshold
    #using boxplots
    #Feature extraction for each activity
    prop = []
    for j in range(4):
        idx = np.where(Y == (j+1))[0]
        properties = wave_eval(AP[idx],CC[idx],ML[idx],Y[idx])
        prop.append(properties[:,0:2])
    
    #Finding boxplots' whiskers
    #for h_AP
    lw_hap, uw_hap = boxplotter([prop[0][:,0],prop[1][:,0],prop[2][:,0],prop[3][:,0]], 
                            ['h_AP_stop', 'h_AP_walk', 'h_AP_jog', 'h_AP_sprint'], 
                            f'Boxplot h_AP Iteration {i}', 0)

    #For CC_max
    lw_cc, uw_cc = boxplotter([prop[0][:,1], prop[1][:,1], prop[2][:,1], prop[3][:,1]], 
                          ['CC_max_stop', 'CC_max_walk', 'CC_max_jog', 'CC_max_sprint'], 
                          f'Boxplot CC_max Iteration {i}', 0)

    
    #Using whiskers to detect intervals
    # Round center values to nearest 0.05 before making ranges
    a = round_to_step(np.mean([float(uw_cc[0]), float(lw_cc[1])]))
    b = round_to_step(np.mean([float(uw_hap[1]), float(lw_hap[2])]))
    c = round_to_step(np.mean([float(uw_hap[2]), float(lw_hap[3])]))

    tresh1 = decimal_range(a - 0.2, a + 0.2, 0.05)
    tresh2 = decimal_range(b - 0.2, b + 0.2, 0.05)
    tresh3 = decimal_range(c - 0.2, c + 0.2, 0.05)
    
    
    properties = wave_eval(AP,CC,ML,Y)
    
    thresholds_range = {
        "threshold_1":tresh1,
        "threshold_2":tresh2,
        "threshold_3":tresh3
        }
    
    #Trying different treshold combinations to find the one with better accuracy
    best_accuracy = 0
    best_thresholds = None
    
    # Find the best threshold combination
    for t1 in thresholds_range["threshold_1"]:
        for t2 in thresholds_range["threshold_2"]:
            for t3 in thresholds_range["threshold_3"]:
                thresholds = [t1,t2,t3]
                L = np.repeat(prop_eval(properties[:,0],properties[:,1],t1,t2,t3),24)
                conf_matrx = np.round(confusioner(Y[0:len(L)],L)*100).astype(int)
                accuracy = np.mean([conf_matrx[0,0],conf_matrx[1,1],conf_matrx[2,2],conf_matrx[3,3]])
                
                if accuracy > best_accuracy and conf_matrx[0,0]>0.9 and conf_matrx[1,1]>0.9 and conf_matrx[2,2]>0.9 and conf_matrx[3,3]>0.9:
                    best_accuracy = accuracy
                    best_thresholds = thresholds
    
    if best_thresholds == None:
        print(f'Iteration{i} is useless')
        continue 
    t1 = best_thresholds[0]
    t2 = best_thresholds[1]
    t3 = best_thresholds[2]
    print(f'iteration {i}, best thresholds: {t1} {t2} {t3}')
    # Salva i best threshold
    best_thresholds_list.append(best_thresholds)

    #Train analysis with selected tresholds
    L = np.repeat(prop_eval(properties[:,0],properties[:,1],t1,t2,t3),24)
    Y = Y[0:len(L)]
    # conf_matrx = np.round(confusioner(Y,L)*100).astype(int)
    # plot_conf(conf_matrx, f'Train Confusion Matrix {i}')
    
    true_label_train.append(Y)
    eval_label_train.append(L)
    
    #%% tresholds application on test files
    
    ap = test[:,0]
    cc = test[:,1]
    ml = test[:,2]
    y = test[:,3]
    properties = wave_eval(ap,cc,ml,y)
    l = np.repeat(prop_eval(properties[:,0],properties[:,1],t1,t2,t3),24)
    y = y[0:len(l)]
    # conf_matrx = np.round(confusioner(y,l)*100).astype(int)
    # plot_conf(conf_matrx, f'Test Confusion Matrix {i}')
    
    true_label_test.append(y)
    eval_label_test.append(l)
    
    trans_values = np.where((y == 5) | (y == 6))
    l[trans_values] = 5
    
    #%% Walking evaluation
    # stepcounting method not use in the final version as the performance obtained with the cadence method were better
    
    W_steps = train[0,4]
    J_steps = train[1,4]
    R_steps = train[2,4]
    
    idx = np.where(Y == 2)
    Walk_train = AP[idx]
    idx = np.where(Y == 3)
    Jog_train = AP[idx]
    idx = np.where(Y == 4)
    Run_train = AP[idx]
    
    thresh_w = thresh_calibrate(Walk_train,W_steps)
    thresh_j = thresh_calibrate(Jog_train,J_steps)
    thresh_r = thresh_calibrate(Run_train,R_steps,1)
    
    # thresh_w = 1.2
    # thresh_j = 1.0
    # thresh_r = 1.6
    
    print(f'Ideal stepcounting threhsolds: {thresh_w} {thresh_j} {thresh_r}')
    # Saves ideal stepcounting thresholds
    stepcounting_thresholds_list.append([thresh_w, thresh_j, thresh_r])
    # thresh_w = 1.6
    # thresh_j = 1.6
    
    #%% Application on the gold standard
    w_steps = test[0,4]
    j_steps = test[1,4]
    r_steps = test[2,4]
    
    idx = np.where(y == 2)
    Walk_test = ap[idx]
    idx = np.where(y == 3)
    Jog_test = ap[idx]
    idx = np.where(y == 4)
    Run_test = ap[idx]
    
    w = stepcounter_thresh(Walk_test,thresh_w)
    err_walk_train = err_walk_train + w-w_steps
    steps_walk_train = steps_walk_train + w_steps
    j = stepcounter_thresh(Jog_test,thresh_j)
    err_jog_train = err_jog_train + j-j_steps
    steps_jog_train = steps_jog_train + j_steps
    r = stepcounter_thresh(Run_test,thresh_r,0,1)
    err_run_train = err_run_train + r-r_steps
    steps_run_train = steps_run_train + r_steps
    
    
    # #%% Application on the Test database
    
    # idx = np.where(l == 2)
    # Walk_test = ap[idx]
    # idx = np.where(l == 3)
    # Jog_test = ap[idx]
    # idx = np.where(l == 4)
    # Run_test = ap[idx]
    
    # w = stepcounter_thresh(Walk_test,thresh_w)
    # err_walk_test = err_walk_test + w-w_steps
    # steps_walk_test = steps_walk_test + w_steps
    # j = stepcounter_thresh(Jog_test,thresh_j)
    # err_jog_test = err_jog_test + j-j_steps
    # steps_jog_test = steps_jog_test + j_steps
    # r = stepcounter_thresh(Run_test,thresh_r,0,1)
    # err_run_test = err_run_test + r-r_steps
    # steps_run_test = steps_run_test + r_steps
    
# average best thresholds
mean_best = np.mean(best_thresholds_list, axis=0)
print(f'\nMean Best Thresholds: {mean_best[0]:.2f}, {mean_best[1]:.2f}, {mean_best[2]:.2f}')

# average ideal stepcounting thresholds
mean_step = np.mean(stepcounting_thresholds_list, axis=0)
print(f'Mean Ideal Stepcounting Thresholds: {mean_step[0]:.2f}, {mean_step[1]:.2f}, {mean_step[2]:.2f}')

full_err_walk_train = (err_walk_train/steps_walk_train)*100
print(f'Train walking error: {full_err_walk_train}')
full_err_jog_train = (err_jog_train/steps_jog_train)*100
print(f'Train jogging error: {full_err_jog_train}')
full_err_run_train = (err_run_train/steps_run_train)*100
print(f'Train sprinting error: {full_err_run_train}')

# full_err_walk_test = (err_walk_test/steps_walk_test)*100
# print(f'Test walking error: {full_err_walk_test}')
# full_err_jog_test = (err_jog_test/steps_jog_test)*100
# print(f'Test jogging error: {full_err_jog_test}')
# full_err_run_test = (err_run_test/steps_run_test)*100
# print(f'Test sprinting error: {full_err_run_test}')
    
    
    


true_label_train = np.hstack(true_label_train)
eval_label_train = np.hstack(eval_label_train)


true_label_test = np.hstack(true_label_test)
eval_label_test = np.hstack(eval_label_test)

true_label_test = true_label_test.astype(float)
eval_label_test = eval_label_test.astype(float)
flags = np.zeros((len(eval_label_test)))

for i in range(12, len(eval_label_test)-12):
    if eval_label_test[i] != eval_label_test[i+1]:
        flags[i-12:i+12] = np.ones((24))
        
conf_matrx_test = np.round(confusioner(true_label_test,eval_label_test,flags)*100).astype(int)
plot_conf(conf_matrx_test, 'Conf Matrix')


#%% saving treshold e stepcounting
results = {
    'Mean Best Threshold 1': [mean_best[0]],
    'Mean Best Threshold 2': [mean_best[1]],
    'Mean Best Threshold 3': [mean_best[2]],
    'Mean Stepcount Walk': [mean_step[0]],
    'Mean Stepcount Jog': [mean_step[1]],
    'Mean Stepcount Run': [mean_step[2]],
    'Gold standard Error Walk (%)': [full_err_walk_train],
    'Gold standard Error Jog (%)': [full_err_jog_train],
    'Gold standard Error Run (%)': [full_err_run_train],
    # 'Test Error Walk (%)': [full_err_walk_test],
    # 'Test Error Jog (%)': [full_err_jog_test],
    # 'Test Error Run (%)': [full_err_run_test]
}


# Ottieni il nome dell'atleta (es. '000')
athlete_id = os.path.basename(athl_dir)

# Crea path completo della cartella dove salvare i risultati
results_dir = os.path.join(athl_dir, f"{athlete_id}_results")

# Crea la cartella se non esiste
os.makedirs(results_dir, exist_ok=True)

# Costruisci il path del file
filename = os.path.join(results_dir, f"threshold_results_{athlete_id}.xlsx")


# Salva il file Excel
df_results = pd.DataFrame(results)
df_results.to_excel(filename, index=False)

print(f"File 'threshold_results' saved in: {filename}")

#%% saving true label e eval label

# Creates DataFrame 
df_labels = pd.DataFrame({
    'True Label Test': true_label_test,
    'Eval Label Test': eval_label_test,
})

# file path 
filename = os.path.join(results_dir, f"labels_comparison_{athlete_id}.xlsx")
# Saving on Excel
df_labels.to_excel(filename, index=False)

print(f"File 'labels_comparison' saved in: {filename}")
