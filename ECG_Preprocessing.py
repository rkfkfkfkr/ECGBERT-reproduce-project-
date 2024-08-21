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
    return record.p_signal, record.fs

def load_dataset(dataset_path, file_extension):
    ecg_signals = []
    patient_ids = []
    
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(file_extension):
            file_path = os.path.join(dataset_path, file_name)
            ecg_data, fs = load_ecg_data(file_path)
            ecg_signals.append((ecg_data, fs))
            patient_ids.append(file_name.split('_')[0])
    
    return ecg_signals, patient_ids

def load_split_signals(dir_path, prefix):
    
    file_path = os.path.join(dir_path, f'{prefix}_split_signals.pkl')
    print(file_path)
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def split_data(signals, patient_ids, test_size=0.2):
    unique_ids = list(set(patient_ids))
    train_ids, val_ids = train_test_split(unique_ids, test_size=test_size, random_state=42)
    
    train_signals = [signals[i] for i in range(len(signals)) if patient_ids[i] in train_ids]
    val_signals = [signals[i] for i in range(len(signals)) if patient_ids[i] in val_ids]
    
    return train_signals, val_signals

def check_missing_values(signal, stage):
    if np.any(np.isnan(signal)):
        print(f"Missing values detected after {stage}")
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

    baseline_removed_signal = np.array(Parallel(n_jobs=-1)(delayed(process_lead)(signal[:, i]) for i in range(signal.shape[1]))).T
    Missingg_value = check_missing_values(baseline_removed_signal, "remove_baseline_wander")
    return baseline_removed_signal, Missingg_value

def preprocess_ecg_signal(signal, fs):
    filtered_signal = bandstop_filter(signal, fs)
    cleaned_signal,Missingg_value  = remove_baseline_wander(filtered_signal, fs)
    return cleaned_signal, Missingg_value

def process_signals(signals):
    # 각 신호를 병렬로 처리
    #processed_signals = Parallel(n_jobs=-1)(delayed(preprocess_ecg_signal)(signal, fs) for signal, fs in signals)
    
    # 결측치가 없는 신호만 선택
    #return [cleaned_signal for cleaned_signal, Missingg_value in processed_signals if not Missingg_value]
    
    processed_signals = []
    signals_without_MV = []
    
    for idx, (signal, fs) in enumerate(signals):
        cleaned_signal, Missingg_value = preprocess_ecg_signal(signal, fs)
        if not Missingg_value:
            processed_signals.append(cleaned_signal)
            signals_without_MV.append((signal, fs))
        else:
            print(idx)
            
    return processed_signals, signals_without_MV
            

def save_processed_data(signals, output_dir, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f'{prefix}_processed_signals.pkl'), 'wb') as f:
        pickle.dump(signals, f)

def save_split_data(signals, output_dir, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f'{prefix}_split_signals.pkl'), 'wb') as f:
        pickle.dump(signals, f)

def ecg_preprocess(dataset_path, save_dir, prefix, file_extension):
    
    ecg_signals, patient_ids = load_dataset(dataset_path, file_extension)
    print(f'Loaded {len(ecg_signals)} ECG signals')

    train_signals, val_signals = split_data(ecg_signals, patient_ids)
    print(f'Split into {len(train_signals)} training and {len(val_signals)} validation signals')
    
    #save_split_data(train_signals, save_dir, f'{prefix}_train')
    #save_split_data(val_signals, save_dir, f'{prefix}_val')
    #print(f'Save {prefix} train, val raw signals')

    #train_signals = load_split_signals(save_dir, f'{prefix}_train')
    #val_signals = load_split_signals(save_dir, f'{prefix}_val')

    #train_signals_processed = Parallel(n_jobs=-1)(delayed(preprocess_ecg_signal)(signal, fs) for signal, fs in train_signals)
    #val_signals_processed = Parallel(n_jobs=-1)(delayed(preprocess_ecg_signal)(signal, fs) for signal, fs in val_signals)
    
    processed_train_signals, train_signals_without_MV = process_signals(train_signals)
    processed_val_signals, val_signals_without_MV = process_signals(val_signals)
    
    save_split_data(train_signals_without_MV, save_dir, f'{prefix}_train')
    save_split_data(val_signals_without_MV, save_dir, f'{prefix}_val')
    print(f'Save {prefix} train : {len(train_signals_without_MV)}, val : {len(val_signals_without_MV)} split signals')
    
    print('ECG preprocessing done')

    save_processed_data(processed_train_signals, save_dir, f'{prefix}_train')
    save_processed_data(processed_val_signals, save_dir, f'{prefix}_val')
    '''
    # not split in preprocessing
    ecg_signals, _ = load_dataset(dataset_path, file_extension)
    print(f'Loaded {len(ecg_signals)} ECG signals')
    
    preprocessed_signals = Parallel(n_jobs=-1)(delayed(preprocess_ecg_signal)(signal, fs) for signal, fs in ecg_signals)
    print('ECG preprocessing done')
    
    save_processed_data(preprocessed_signals, save_dir, f'{prefix}')
    '''
if __name__ == '__main__':
    dataset_paths = {
        'cpsc': 'D:/data/ECGBERT/org_data/pre-train/cpsc/',
        'georgia': 'D:/data/ECGBERT/org_data/pre-train/georgia/',
        'ptb_xl': 'D:/data/ECGBERT/org_data/pre-train/ptb_xl/'
    }
    
    save_dir = 'D:/data/ECGBERT/for_git3/preprocessing/ECG_preprocessing/'
    
    for prefix, dataset_path in dataset_paths.items():
        file_extension = '.mat' if prefix != 'ptb_xl' else '.dat'
        ecg_preprocess(dataset_path, save_dir, prefix, file_extension)
        print(f'{prefix} preprocessing done')