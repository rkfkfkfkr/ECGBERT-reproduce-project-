import wfdb
import numpy as np
import pywt
from scipy.signal import butter, sosfilt, medfilt, filtfilt
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import pickle
import os

def load_ecg_data(record_path):
    record = wfdb.rdrecord(record_path[:-4])
    annotation = wfdb.rdann(record_path[:-4], 'atr')
    
    # p_signal의 길이와 동일한 길이의 labels_list 초기화 (0으로 채움)
    label = np.zeros(len(record.p_signal), dtype=int)
    
    # AFIB 구간을 1로 표시
    afib_indices = np.where(np.isin(annotation.symbol, ['~']))[0]
    for idx in afib_indices:
        start_idx = annotation.sample[idx]
        end_idx = annotation.sample[idx + 1] if idx + 1 < len(annotation.sample) else len(label)
        label[start_idx:end_idx] = 1
    
    return record.p_signal, record.fs, label

def load_dataset(dataset_path, file_extension):
    ecg_signals = []
    patient_ids = []
    labels = []
    
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(file_extension):
            file_path = os.path.join(dataset_path, file_name)
            ecg_data, fs, label = load_ecg_data(file_path)
            ecg_signals.append((ecg_data, fs))
            patient_ids.append(file_name.split('_')[0])
            labels.append(label)
    
    return ecg_signals, patient_ids, labels

def load_split_signals(dir_path, prefix):
    
    file_path = os.path.join(dir_path, f'{prefix}_split_signals.pkl')
    print(file_path)
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def split_data(signals, patient_ids, labels, test_size=0.2):
    unique_ids = list(set(patient_ids))
    train_ids, val_ids = train_test_split(unique_ids, test_size=test_size, random_state=42)
    
    train_signals = [signals[i] for i in range(len(signals)) if patient_ids[i] in train_ids]
    val_signals = [signals[i] for i in range(len(signals)) if patient_ids[i] in val_ids]
    
    # label
    train_labels = [labels[i] for i in range(len(labels)) if patient_ids[i] in train_ids]
    val_labels = [labels[i] for i in range(len(labels)) if patient_ids[i] in val_ids]
    
    return train_signals, val_signals, train_labels, val_labels

def check_missing_values(signal, stage):
    if np.any(np.isnan(signal)):
        #print(f"Missing values detected after {stage}")
        Missingg_value = True
    else:
        Missingg_value = False
        
    return Missingg_value

def bandstop_filter(signal, fs, lowcut=50.0, highcut=60.0, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b,a = butter(order, [low, high], btype='bandstop', fs=fs)
    #sos = butter(order, [low, high], btype='bandstop', output='sos')

    # 각 리드에 대해 독립적으로 필터링을 적용
    #filtered_signal = np.array([sosfilt(sos, signal[:, i]) for i in range(signal.shape[1])]).T
    filtered_signal = np.array([filtfilt(b,a, signal[:, i]) for i in range(signal.shape[1])]).T
    
    return filtered_signal

def remove_baseline_wander(signal, fs, wavelet='db4', level=2):
    baseline_removed_signal = np.zeros_like(signal)

    def process_lead(lead_signal):
        
        coeffs = pywt.wavedec(lead_signal, wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        baseline_wander_signal = pywt.waverec(coeffs, wavelet)
        
        if len(baseline_wander_signal) > len(lead_signal):
            baseline_wander_signal = baseline_wander_signal[:len(lead_signal)]
        elif len(baseline_wander_signal) < len(lead_signal):
            baseline_wander_signal = np.pad(baseline_wander_signal, (0, len(lead_signal) - len(baseline_wander_signal)), 'constant')
        
        remove_signal = lead_signal - baseline_wander_signal
        #lowpassed = medfilt(remove_signal, kernel_size=fs + 1)
        return remove_signal #- lowpassed

    # 각 리드를 순차적으로 처리
    for i in range(signal.shape[1]):
        baseline_removed_signal[:, i] = process_lead(signal[:, i])
    
    Missingg_value = check_missing_values(baseline_removed_signal, "remove_baseline_wander")
    return baseline_removed_signal, Missingg_value

def preprocess_ecg_signal(signal, fs):
    filtered_signal = bandstop_filter(signal, fs)
    cleaned_signal,Missingg_value  = remove_baseline_wander(filtered_signal, fs)
    return cleaned_signal, Missingg_value

def process_signals(signals, labels):
    
    processed_signals = []
    split_labels = []
    
    for idx, (signal, fs) in enumerate(signals):
        cleaned_signal, Missingg_value = preprocess_ecg_signal(signal, fs)
        if not Missingg_value:
            processed_signals.append(cleaned_signal)
            split_labels.append(labels[idx])
        else:
            print(idx)
            
    return processed_signals, split_labels
            
def save_pkl_data(save_dir, file_name, save_pkl):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_pkl, f)

def ecg_preprocess(dataset_path, save_dir, downstrem_task, file_extension):
    
    ecg_signals, patient_ids, labels = load_dataset(dataset_path, file_extension)

    train_signals, val_signals, train_labels, val_labels = split_data(ecg_signals, patient_ids, labels)

    processed_train_signals, train_labels = process_signals(train_signals, train_labels)
    processed_val_signals, val_labels = process_signals(val_signals, val_labels)
    
    save_pkl_data(save_dir, f'{downstrem_task}_train_processed_signals.pkl', processed_train_signals)
    save_pkl_data(save_dir, f'{downstrem_task}_val_processed_signals.pkl', processed_val_signals)
    
    save_pkl_data(save_dir, f'{downstrem_task}_train_labels.pkl', train_labels)
    save_pkl_data(save_dir, f'{downstrem_task}_val_labels.pkl', val_labels)

import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def ECG_Preprocessing(dataset_paths, dir):
    
    for downstrem_task, dataset_path, file_extension in dataset_paths:
        
        save_dir = os.path.join(dir, f'{downstrem_task}/ECG_Preprocessing')
        ecg_preprocess(dataset_path, save_dir, downstrem_task, file_extension)
        logger.info(f'{downstrem_task} preprocessing done')
        
# dataset_paths = [ [prefix, dataset_path, file_extension], ...  ]