# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:06:37 2025

@author: isotta
"""


import os
import pandas as pd
import seaborn as sns
import numpy as np
import random
import pywt
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext, ROUND_HALF_UP


sr = 12.5

def loader(name):
    base_dir = os.getcwd()
    directory = os.path.join(base_dir, name)
    test = os.path.join(directory, 'Test')
    train = os.path.join(directory, 'Train')
    return base_dir, directory, test, train

def database_create (directory):
    csv_files = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            csv_files.append(file_path)
            
    arrays_other = []
    arrays_stop = []
    arrays_walk = []
    arrays_jog = []
    arrays_run = []
        
    names_other = []
    names_stop = []
    names_walk = []
    names_jog = []
    names_run = []
    
    for file_path in csv_files:
        try:
            file_name = os.path.basename(file_path)
            df = pd.read_csv(file_path, header=None)
            df = df.values
            
            l = len(df[:, 0])
            signal = np.zeros((l, 5))
            act = df[0, 3]
            signal[:, 0:3] = df[:, 0:3]
            signal[0, 4] = df[0, 4]

            if act == 0:
                signal[:, 3] = np.zeros(l)
                arrays_other.append(signal)
                names_other.append(file_name)
            elif act == 1:
                signal[:, 3] = np.ones(l)
                arrays_stop.append(signal)
                names_stop.append(file_name)
            elif act == 2:
                signal[:, 3] = np.ones(l) * 2
                arrays_walk.append(signal)
                names_walk.append(file_name)
            elif act == 3:
                signal[:, 3] = np.ones(l) * 3
                arrays_jog.append(signal)
                names_jog.append(file_name)
            elif act == 4:
                signal[:, 3] = np.ones(l) * 4
                arrays_run.append(signal)
                names_run.append(file_name)
                    
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            
    arrays = [arrays_other, arrays_stop, arrays_walk, arrays_jog, arrays_run]
    names = [names_other, names_stop, names_walk, names_jog, names_run]
    
    return arrays, names

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

def mid_high (data):
    m = max(data[39:59])
    h = max(data[89:99])
    return m, h

def whiskers(array):
    """
    Compute the lower and upper whiskers of a 1D numerical array using the IQR method.

    Parameters:
        array (list or np.array): Input numerical data.

    Returns:
        lower_whisker (float): Lower whisker value.
        upper_whisker (float): Upper whisker value.
    """
    array = np.array(array)
    array = array[~np.isnan(array)]  # Remove NaNs if present

    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Actual whiskers are the most extreme values within these bounds
    lower_whisker = array[array >= lower_bound].min()
    upper_whisker = array[array <= upper_bound].max()

    return lower_whisker, upper_whisker

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
            y[pos] = 100
            l[pos] = 100

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

def stack_n(data,n,flag=0):
    d = []
    s =  0
    if flag == 0:
        for i in range(n):
            d.append(data[i])
            s+=data[i][0,4]
    else:
        for i in range(n):
            d.append(data[i])
            
    d = np.vstack(d)
    
    return d,s

def step_number_filter(data,n):
    d = []
    for i in range(len(data)):
        if data[i][0,4] > n:
            d.append(data[i])
    return d
            
def thresholds_range_calc(properties):
    ps = properties[properties[:,2] == 1]
    pw = properties[properties[:,2] == 2]
    pj = properties[properties[:,2] == 3]
    pr = properties[properties[:,2] == 4]
            
    ls,us = whiskers(ps[:,1])
    lw,_ = whiskers(pw[:,1])
    _,uw = whiskers(pw[:,0])
    lj,uj = whiskers(pj[:,0])
    lr,_ = whiskers(pr[:,0])
            
    #Uso le whisker per identificare gli intervalli
    # Round center values to nearest 0.05 before making ranges
    a = round_to_step(np.mean([float(us), float(lw)]))
    b = round_to_step(np.mean([float(uw), float(lj)]))
    c = round_to_step(np.mean([float(uj), float(lr)]))
    tresh1 = decimal_range(a - 0.2, a + 0.2, 0.05)
    tresh2 = decimal_range(b - 0.2, b + 0.2, 0.05)
    tresh3 = decimal_range(c - 0.2, c + 0.2, 0.05)
            
    thresholds_range = {
    "threshold_1":tresh1,
    "threshold_2":tresh2,
    "threshold_3":tresh3
    }
    
    return thresholds_range

def best_thresholds_calc(properties):
    
    thresholds_range = thresholds_range_calc(properties)
    best_accuracy = 0
    best_thresholds = None
            
    # Find the best threshold combination
    for t1 in thresholds_range["threshold_1"]:
        for t2 in thresholds_range["threshold_2"]:
            for t3 in thresholds_range["threshold_3"]:
                thresholds = [t1,t2,t3]
                l = np.repeat(prop_eval(properties[:,0],properties[:,1],t1,t2,t3),24)
            
                conf_matrx = np.round(confusioner(y[0:len(l)],l)*100).astype(int)
                accuracy = np.mean([conf_matrx[0,0],conf_matrx[1,1],conf_matrx[2,2],conf_matrx[3,3]])
            
                if accuracy > best_accuracy and conf_matrx[0,0]>0.5 and conf_matrx[1,1]>0.5 and conf_matrx[2,2]>0.5 and conf_matrx[3,3]>0.5:
                    best_accuracy = accuracy
                    best_thresholds = thresholds
    return best_thresholds, best_accuracy
 
def database_test(s,w,j,r):
    
    stop = []
    for i in range(len(s)):
        interval = s[i]
        if i == 0:
            interval1 = interval
            interval2 = interval
            flags1 = np.ones((interval1.shape[0], 1))
            flags2 = np.zeros((interval1.shape[0], 1))
            interval = np.vstack([interval1, interval2])
            flags = np.vstack([flags1, flags2])
        elif i == len(s) - 1:
            interval1 = interval
            interval2 = interval
            flags1 = np.zeros((interval1.shape[0], 1))
            flags2 = np.ones((interval1.shape[0], 1))
            interval = np.vstack([interval1, interval2])
            flags = np.vstack([flags1, flags2])
        else:
            flags = np.zeros((interval.shape[0], 1))
    
        interval = np.hstack((interval, flags))
        stop.append(interval)
    stop = np.vstack(stop)
    
    walk = []
    steps_w = 0
    len_w = 0
    for i in range(len(w)):
        interval = w[i]
        if i == 0:
            interval1 = interval
            interval2 = interval
            flags1 = np.ones((interval1.shape[0], 1))
            flags2 = np.zeros((interval1.shape[0], 1))
            interval = np.vstack([interval1, interval2])
            flags = np.vstack([flags1, flags2])
            steps_w += interval[0,4]
            len_w += len(interval[:,0])
        elif i == len(w) - 1:
            interval1 = interval
            interval2 = interval
            flags1 = np.zeros((interval1.shape[0], 1))
            flags2 = np.ones((interval1.shape[0], 1))
            interval = np.vstack([interval1, interval2])
            flags = np.vstack([flags1, flags2])
            steps_w += interval[0,4]
            len_w += len(interval[:,0])
        else:
            flags = np.zeros((interval.shape[0], 1))
            steps_w += interval[0,4]
            len_w += len(interval[:,0])
    
        interval = np.hstack((interval, flags))
        walk.append(interval)
    walk = np.vstack(walk)
    
    jog = []
    steps_j = 0
    len_j = 0
    for i in range(len(j)):
        interval = j[i]
        if i == 0:
            interval1 = interval
            interval2 = interval
            flags1 = np.ones((interval1.shape[0], 1))
            flags2 = np.zeros((interval1.shape[0], 1))
            interval = np.vstack([interval1, interval2])
            flags = np.vstack([flags1, flags2])
            steps_j += interval[0,4]
            len_j += len(interval[:,0])
        elif i == len(j) - 1:
            interval1 = interval
            interval2 = interval
            flags1 = np.zeros((interval1.shape[0], 1))
            flags2 = np.ones((interval1.shape[0], 1))
            interval = np.vstack([interval1, interval2])
            flags = np.vstack([flags1, flags2])
            steps_j += interval[0,4]
            len_j += len(interval[:,0])
        else:
            flags = np.zeros((interval.shape[0], 1))
            steps_j += interval[0,4]
            len_j += len(interval[:,0])
    
        interval = np.hstack((interval, flags))
        jog.append(interval)
    jog = np.vstack(jog)
    
    run = []
    steps_r = 0
    len_r = 0
    for i in range(len(r)):
        interval = r[i]
        if i == 0:
            interval1 = interval
            interval2 = interval
            flags1 = np.ones((interval1.shape[0], 1))
            flags2 = np.zeros((interval1.shape[0], 1))
            interval = np.vstack([interval1, interval2])
            flags = np.vstack([flags1, flags2])
            steps_r += interval[0,4]
            len_r += len(interval[:,0])
        elif i == len(r) - 1:
            interval1 = interval
            interval2 = interval
            flags1 = np.zeros((interval1.shape[0], 1))
            flags2 = np.ones((interval1.shape[0], 1))
            interval = np.vstack([interval1, interval2])
            flags = np.vstack([flags1, flags2])
            steps_r += interval[0,4]
            len_r += len(interval[:,0])
        else:
            flags = np.zeros((interval.shape[0], 1))
            steps_r += interval[0,4]
            len_r += len(interval[:,0])
    
        interval = np.hstack((interval, flags))
        run.append(interval)
    run = np.vstack(run)
    
    data = np.vstack((stop, walk, jog, run))
    steps = [steps_w, steps_j, steps_r]
    lens = [len_w, len_j, len_r]
    return data, steps, lens
            
def plot_conf(conf_mtrx,title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mtrx, annot=True, fmt='g', cmap='GnBu', xticklabels=['Stop', 'Walk', 'Jog', 'Sprint'], yticklabels=['Stop', 'Walk', 'Jog', 'Sprint'])

    # Adding labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()         
            
        
#%% Main

base_dir, Dir , Test, Train = loader('001')
arrays, names = database_create(Dir)
ideal_t = [1.14, 0.55, 1.85]

stop = arrays[1]
walk = arrays[2]
jog = arrays[3]
run = arrays[4]

print(f'stop = {len(stop)}')
print(f'stop = {len(walk)}')
print(f'stop = {len(jog)}')
print(f'stop = {len(run)}')

# stop = stop*3
# walk = walk*3
# jog = jog*3
# run = run*3

# test,steps = database_test(stop,walk,jog,run)
# AP = test[:,0]
# CC = test[:,1]
# ML = test[:,2]
# Y = test[:,3]
# prop_test = wave_eval(AP,CC,ML,Y)
# flags = test[:,5]
t = []

l_tot = []
y_tot = []
flags_tot = []

err_walk_train = 0
steps_walk_train = 0
err_jog_train = 0
steps_jog_train = 0
err_run_train = 0
steps_run_train = 0

err_walk_test = 0
steps_walk_test = 0
err_jog_test = 0
steps_jog_test = 0
err_run_test = 0
steps_run_test = 0

a = 4
b = 2
flag_w = 0
flag_j = 0
flag_r = 0
for i in range(50):
    print(f'full try {i}')
    
    S = stop
    W = walk
    J = jog
    R = run
    
    
    random.shuffle(S)
    s = S[:a]
    s_ = S[a:]

    random.shuffle(W)
    w = W[:a]
    w_ = W[a:]
      
    random.shuffle(J)
    j = J[:a]
    j_ = J[a:]

    random.shuffle(R)
    r = R[:b]
    r_ = R[b:]
         
    
    test,steps,lens = database_test(s_,w_,j_,r_)
    AP = test[:,0]
    CC = test[:,1]
    ML = test[:,2]
    Y = test[:,3]
    prop_test = wave_eval(AP,CC,ML,Y)
    flags = test[:,5]

    random.shuffle(s)
    random.shuffle(w)
    random.shuffle(j)
    random.shuffle(r)

    Stop,_ = stack_n(s,4,1)
    Walk, w_steps = stack_n(w,4)
    Jog, j_steps = stack_n(j,4)
    Run, r_steps = stack_n(r,2)

                    
    data = [Stop,
            Walk,
            Jog,
            Run]
    data = np.vstack(data)

    ap = data[:,0]
    cc = data[:,1]
    ml = data[:,2]
    y = data[:,3]
                
    properties = wave_eval(ap,cc,ml,y)                        
    best_thresholds,_ = best_thresholds_calc(properties)
            
    if best_thresholds == None:
        continue
            
    print(f'Best Thresholds: {best_thresholds}')
    T1 = round(best_thresholds[0],2)
    T2 = round(best_thresholds[1],2)
    T3 = round(best_thresholds[2],2)
        
    time_w = len(Walk[:,0])/(sr*60)
    c_w = w_steps/time_w
    CW = round(c_w,2)
    time_j = len(Jog[:,0])/(sr*60)
    c_j = j_steps/time_j
    CJ = round(c_j,2)
    time_r = len(Run[:,0])/(sr*60)
    c_r = r_steps/time_r
    CR = round(c_r,2)
        
    # # Boxplot of the three threshold arrays
    # plt.figure(figsize=(8, 6))
    # plt.boxplot([T1, T2, T3], tick_labels=['t1 (Stop-Walk)', 't2 (Walk-Jog)', 't3 (Jog-Run)'])
    # plt.title('Boxplot of Threshold Arrays')
    # plt.ylabel('Value')
    # plt.grid(True)
    # plt.show()
    
    t.append([T1, T2, T3, CW, CJ, CR])
    
    l = np.repeat(prop_eval(prop_test[:,0],prop_test[:,1],T1,T2,T3),24)
    y_temp = Y[:len(l)]
    f = flags[:len(l)]
    conf_matrx = np.round(confusioner(y_temp,l,f)*100).astype(int)
    plot_conf(conf_matrx, f'Test Confusion Matrix {i}')
    
    for k in range(len(y_temp)):
        if f[k] == 1:
            y_temp[k] = 100
            l[k] = 100
    
    l_tot.append(l)
    y_tot.append(y_temp)
    flags_tot.append(f)
            
    # Step counting
    
    lens[0] = np.sum(y_temp == 2)
    lens[1] = np.sum(y_temp == 3)
    lens[2] = np.sum(y_temp == 4)
            
    time_w = lens[0]/(sr*60)
    err_walk_train += steps[0] - (CW*time_w)
    steps_walk_train += steps[0]
    
    time_j = lens[1]/(sr*60)
    err_jog_train += steps[1] - (CJ*time_j)
    steps_jog_train += steps[1]
    
    time_r = lens[2]/(sr*60)
    err_run_train += steps[2] - (CR*time_r)
    steps_run_train += steps[2]
    
    # Combined
    
    lens[0] = np.sum(l == 2)
    lens[1] = np.sum(l == 3)
    lens[2] = np.sum(l == 4)
            
    time_w = lens[0]/(sr*60)
    err_walk_test += steps[0] - (CW*time_w)
    steps_walk_test += steps[0]
    
    time_j = lens[1]/(sr*60)
    err_jog_test += steps[1] - (CJ*time_j)
    steps_jog_test += steps[1]
    
    time_r = lens[2]/(sr*60)
    err_run_test += steps[2] - (CR*time_r)
    steps_run_test += steps[2]
    
    
t = np.vstack(t)
plt.figure(figsize=(8, 6))
plt.boxplot([t[:,0], t[:,1], t[:,2]], tick_labels=['t1 (Stop-Walk)', 't2 (Walk-Jog)', 't3 (Jog-Run)'])
plt.title('Boxplot of Threshold Arrays')
plt.ylabel('Value')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot([t[:,3], t[:,4], t[:,5]], tick_labels=['Cw', 'Cj', 'Cr'])
plt.title('Boxplot of Cadences')
plt.ylabel('Value')
plt.grid(True)
plt.show()

l_tot = np.hstack(l_tot)
y_tot = np.hstack(y_tot)
flags_tot = np.hstack(flags_tot)
conf_matrx = np.round(confusioner(y_tot,l_tot,flags_tot)*100).astype(int)
plot_conf(conf_matrx, f'Tot Performances')

train_err_w = round((err_walk_train/steps_walk_train)*100,2)
train_err_j = round((err_jog_train/steps_jog_train)*100,2)
train_err_r = round((err_run_train/steps_run_train)*100,2)

test_err_w = round((err_walk_test/steps_walk_test)*100,2)
test_err_j = round((err_jog_test/steps_jog_test)*100,2)
test_err_r = round((err_run_test/steps_run_test)*100,2)

print('-------------------------------------')
print(f"Walk train error: {train_err_w}%")
print(f"Jog train error: {train_err_j}%")
print(f"Sprint train error: {train_err_r}%")

print('-------------------------------------')
print(f"Walk test error: {test_err_w}%")
print(f"Jog test error: {test_err_j}%")
print(f"Sprint test error: {test_err_r}%")