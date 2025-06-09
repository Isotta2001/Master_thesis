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


SAMPLING_RATE = 12.5
#SAMPLING_RATE = 200

#%% Functions
#Funding directory
def loader(name):
    base_dir = os.getcwd()
    directory = os.path.join(base_dir, name)
    test = os.path.join(directory, 'Test')
    train = os.path.join(directory, 'Train')
    return base_dir, directory, test, train

#Function to get file csv in the folder
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

# Function to get features 
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

#Function to get wavelet features
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

# Function to get high and medium frequency peaks
def mid_high (data):
    m = max(data[39:59])
    h = max(data[89:99])
    return m, h

#Function to get whiskers and print boxplot
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

#Function to get labels array
def prop_eval (x,y,t1,t2,t3):
    l = []
    for i in range(len(x)):
        l.append(act_identifier(x[i],y[i],t1,t2,t3))
    l = np.array(l)
    return l

#Function for activity identification
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

#Function to obtain the confusion matrix
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

base_dir, athl_dir , Test_dir, Train_dir = loader('006')
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

cad_w = []
cad_j = []
cad_r = []
best_thresholds_list = []
stepcounting_thresholds_list = []

for i in range(len(Train_files)): #(len(Train_files)):
    
    print(f'Initiating cycle {i}')
    
    #%% Threshold evaluation
    test = Test_files[i]
    train = Train_files[i]
    
    #label and axis extractions
    AP = train[:,0]
    AP = np.pad(AP, (0, 24), mode='constant', constant_values=0)
    CC = train[:,1]
    CC = np.pad(CC, (0, 24), mode='constant', constant_values=0)
    ML = train[:,2]
    ML = np.pad(ML, (0, 24), mode='constant', constant_values=0)
    Y = train[:,3]
    Y = np.pad(Y, (0, 24), mode='constant', constant_values=0)
    
    #TFinding ideal tresholds ranges using boxplots
    #first features extraction for each activity
    prop = []
    for j in range(4):
        idx = np.where(Y == (j+1))[0]
        properties = wave_eval(AP[idx],CC[idx],ML[idx],Y[idx])
        prop.append(properties[:,0:2])
    
    #boxplots' whiskers
    lw_hap, uw_hap = boxplotter([prop[0][:,0],prop[1][:,0],prop[2][:,0],prop[3][:,0]], 
                            ['h_AP_stop', 'h_AP_walk', 'h_AP_jog', 'h_AP_sprint'], 
                            f'Boxplot h_AP Iteration {i}', 0)

    #For CC_max
    lw_cc, uw_cc = boxplotter([prop[0][:,1], prop[1][:,1], prop[2][:,1], prop[3][:,1]], 
                          ['CC_max_stop', 'CC_max_walk', 'CC_max_jog', 'CC_max_sprint'], 
                          f'Boxplot CC_max Iteration {i}', 0)

    
    #Using whiskers to define tresholds
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
    
    #Finding ideal tresholds combination

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
    # Saves best thresholds
    best_thresholds_list.append(best_thresholds)


    #Ananlysis with identified tresholds
    L = np.repeat(prop_eval(properties[:,0],properties[:,1],t1,t2,t3),24)
    Y = Y[0:len(L)]
    # conf_matrx = np.round(confusioner(Y,L)*100).astype(int)
    # plot_conf(conf_matrx, f'Train Confusion Matrix {i}')
    
    true_label_train.append(Y)
    eval_label_train.append(L)
    
    #%% Application on test files
    
    # if already known i can insert treshold here to speed up the process and avoid the iteration loop processing time
    # t1 = 1.798
    # t2 = 0.908
    # t3 = 2.512
    
    ap = test[:,0]
    cc = test[:,1]
    ml = test[:,2]
    y = test[:,3]
    properties = wave_eval(ap,cc,ml,y)
    l = np.repeat(prop_eval(properties[:,0],properties[:,1],t1,t2,t3),24)
    l_temp = np.zeros((len(y)))
    l_temp[0:len(l)] = l
    l = l_temp
    #y = y[0:len(l)]
    # conf_matrx = np.round(confusioner(y,l)*100).astype(int)
    # plot_conf(conf_matrx, f'Test Confusion Matrix {i}')
    
    true_label_test.append(y)
    eval_label_test.append(l)
    
    trans_values = np.where((y == 5) | (y == 6))
    y[trans_values] = 5
    l[trans_values] = 5
    
    
    #%% Walking evaluation
    
    W_steps = train[0,4]
    J_steps = train[1,4]
    R_steps = train[2,4]
    
    idx = np.where(Y == 2)
    Walk_train = AP[idx]
    idx = np.where(Y == 3)
    Jog_train = AP[idx]
    idx = np.where(Y == 4)
    Run_train = AP[idx]
    
    time_W = len(Walk_train)/(SAMPLING_RATE*60)
    c_w = W_steps/time_W
    time_J = len(Jog_train)/(SAMPLING_RATE*60)
    c_j = J_steps/time_J
    time_R = len(Run_train)/(SAMPLING_RATE*60)
    c_r = R_steps/time_R
    
    #%% Application on the gold standard
    
    w_steps = test[0,4] 
    j_steps = test[1,4] 
    r_steps = test[2,4] 
    
    idx = np.where(y==2)
    walk_test = ap[idx]
    idx = np.where(y==3)
    jog_test = ap[idx]
    idx = np.where(y==4)
    run_test = ap[idx]
    
    w_time = len(walk_test)/(SAMPLING_RATE*60)
    err_walk_train += w_steps - (c_w*w_time)
    steps_walk_train += w_steps
    
    j_time = len(jog_test)/(SAMPLING_RATE*60)
    err_jog_train += j_steps - (c_j*j_time)
    steps_jog_train += j_steps
    
    r_time = len(run_test)/(SAMPLING_RATE*60)
    err_run_train += r_steps - (c_r*r_time)
    steps_run_train += r_steps
    
    
    
    #%% Application on the Test database
    
    idx = np.where(l==2) 
    walk_test = ap[idx]
    idx = np.where(l==3) 
    jog_test = ap[idx]
    idx = np.where(l==4) 
    run_test = ap[idx]
    
    w_time = len(walk_test)/(SAMPLING_RATE*60)
    err_walk_test += w_steps - (c_w*w_time)
    steps_walk_test += w_steps
    
    j_time = len(jog_test)/(SAMPLING_RATE*60)
    err_jog_test += j_steps - (c_j*j_time)
    steps_jog_test += j_steps
    
    r_time = len(run_test)/(SAMPLING_RATE*60)
    err_run_test += r_steps - (c_r*r_time)
    steps_run_test += r_steps
    
    cad_w.append(c_w)
    cad_j.append(c_j)
    cad_r.append(c_r)       
    
    #%% Interval plot
    
    a = 0
    
    if a != 0:
        plt.figure(figsize = (10,7) )
    
        plt.subplot(311)
        plt.plot(ap[0:450], color = 'black')
        plt.title('signal')
        plt.xlabel('Frames')  # x-axis label
        plt.ylabel('g')  # y-axis label for the first subplot
    
        plt.subplot(312)
        plt.plot(y[0:450], color = 'Blue', label = 'Labels')
        plt.axhline(y=1, color='lightblue', linestyle='--', label='Stop')
        plt.axhline(y=2, color='green', linestyle='--', label='Walk')
        plt.axhline(y=3, color='yellow', linestyle='--', label='Jog')
        plt.axhline(y=4, color='orange', linestyle='--', label='Sprint')
        plt.axhline(y=5, color='purple', linestyle='--', label='Transition')
        plt.legend()
        plt.xlabel('Frames')  # x-axis label
        plt.ylabel('Activity')  # y-axis label for the first subplot
        plt.title('Gold Standard')
        plt.ylim(0,6)
    
        plt.subplot(313)
        plt.plot(l[0:450], color = 'red', label = 'Tags')
        plt.axhline(y=1, color='lightblue', linestyle='--', label='Stop')
        plt.axhline(y=2, color='green', linestyle='--', label='Walk')
        plt.axhline(y=3, color='yellow', linestyle='--', label='Jog')
        plt.axhline(y=4, color='orange', linestyle='--', label='Sprint')
        plt.axhline(y=5, color='purple', linestyle='--', label='Transition')
        plt.legend()
        plt.xlabel('Frames')  # x-axis label
        plt.ylabel('Activity')  # y-axis label for the first subplot
        plt.title('Algorithm')
        plt.ylim(0,6)
    
        plt.tight_layout(pad=2.0)
        #plt.subplots_adjust(top=0.95)
        plt.show()


#%% Performances

gold_err_w = round((err_walk_train/steps_walk_train)*100,2)
gold_err_j = round((err_jog_train/steps_jog_train)*100,2)
gold_err_r = round((err_run_train/steps_run_train)*100,2)

test_err_w = round((err_walk_test/steps_walk_test)*100,2)
test_err_j = round((err_jog_test/steps_jog_test)*100,2)
test_err_r = round((err_run_test/steps_run_test)*100,2)

cad_w = np.mean(np.array(cad_w))
cad_j = np.mean(np.array(cad_j))
cad_r = np.mean(np.array(cad_r))

print('-------------------------------------')
print(f"Walk train error: {gold_err_w}")
print(f"Jog train error: {gold_err_j}")
print(f"Sprint train error: {gold_err_r}")

print('-------------------------------------')
print(f"Walk test error: {test_err_w}")
print(f"Jog test error: {test_err_j}")
print(f"Sprint test error: {test_err_r}")

print('-------------------------------------')
print(f"Walk cadence: {cad_w}")
print(f"Jog cadence: {cad_j}")
print(f"Sprint cadence: {cad_r}")


#%% savings

# # athlete ID (es. '000')
# athlete_id = os.path.basename(athl_dir)

# # create folder path to save data
# results_dir = os.path.join(athl_dir, f"{athlete_id}_results")

# # Create folder if non existing
# os.makedirs(results_dir, exist_ok=True)

# # file path
# filename = os.path.join(results_dir, f"threshold_results_{athlete_id}.xlsx")


# # Save on Excel
# df_results = pd.DataFrame(results)
# df_results.to_excel(filename, index=False)

# print(f"File 'threshold_results' salvato in: {filename}")

#%% saving true label e eval label

# # Crea un DataFrame con i dati
# df_labels = pd.DataFrame({
#     'True Label Test': true_label_test,
#     'Eval Label Test': eval_label_test,
# })

# # file path
# filename = os.path.join(results_dir, f"labels_comparison_{athlete_id}.xlsx")
# #Excel on saving
# df_labels.to_excel(filename, index=False)

# print(f"File 'labels_comparison' salvato in: {filename}")