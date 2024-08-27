import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks
import pickle
import os

def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def detect_peaks(signals, fs):
    import neurokit2 as nk
    
    clean_signals = np.array([nk.ecg_clean(signals[:, i], sampling_rate=fs, method='hamilton2002') for i in range(signals.shape[1])]).T
    all_r_peaks = [nk.ecg_peaks(clean_signals[:, i], sampling_rate=fs, method='hamilton2002')[1]["ECG_R_Peaks"] for i in range(signals.shape[1])]
    
    return clean_signals, all_r_peaks

def segment_waveforms(clean_signals, r_peaks, fs):
    max_level = 4
    wavelet = 'db4'
    lead_segments = []

    for lead in range(clean_signals.shape[1]):
        lead_wave_seg = {'p': [], 'qrs': [], 't': [], 'bg': []}
        
        lead_r_peaks = r_peaks[lead]
        
        if len(lead_r_peaks) == 0:
            continue
        
        qrs_start = lead_r_peaks - int(0.05 * fs)
        qrs_end = lead_r_peaks + int(0.05 * fs)
        p_end = qrs_start
        p_start = p_end - int(0.1 * fs)
        t_start = qrs_end
        t_end = t_start + int(0.15 * fs)
        
        qrs_segment = np.clip(np.vstack((qrs_start, qrs_end)).T, 0, clean_signals.shape[0])
        p_segment = np.clip(np.vstack((p_start, p_end)).T, 0, clean_signals.shape[0])
        t_segment = np.clip(np.vstack((t_start, t_end)).T, 0, clean_signals.shape[0])
        
        if len(p_segment) > 0 and len(t_segment) > 0:
            bg_segment = np.array([(0, p_segment[0, 0])] + 
                                  [(t_segment[i, 1], p_segment[i+1, 0]) for i in range(len(t_segment) - 1)] + 
                                  [(t_segment[-1, 1], clean_signals.shape[0])])
        else:
            bg_segment = np.array([(0, clean_signals.shape[0])])
        
        lead_wave_seg['qrs'].append(qrs_segment)
        lead_wave_seg['p'].append(p_segment)
        lead_wave_seg['t'].append(t_segment)
        lead_wave_seg['bg'].append(bg_segment)
        
        lead_segments.append(lead_wave_seg)
        
    return lead_segments

def save_segments(lead_segments, save_dir, prefix):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'{prefix}_segments.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(lead_segments, f)
    
    for wave_type in ['p', 'qrs', 't', 'bg']:
        wave_type_dir = os.path.join(save_dir, wave_type)
        if not os.path.exists(wave_type_dir):
            os.makedirs(wave_type_dir)
        wave_segments = [seg[wave_type] for seg in lead_segments]
        save_path = os.path.join(wave_type_dir, f'{prefix}_{wave_type}_segments.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(wave_segments, f)

def save_pkl_data(save_dir, file_name, save_pkl):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_pkl, f)

def process_and_save_segments(processed_data_dir, save_dir, downstrem_task, fs):
    for suffix in ['train', 'val']:
        prefix = f'{downstrem_task}_{suffix}'
            
        file_path = os.path.join(processed_data_dir, f'{prefix}_processed_signals.pkl')
        signals = load_pkl_data(file_path)

        file_path = os.path.join(processed_data_dir, f'{prefix}_labels.pkl')
        labels = load_pkl_data(file_path)

        seg_data_num = 0
            
        for idx, signal in enumerate(signals):
                
            clean_signals, r_peaks = detect_peaks(signal, fs)
            lead_segments = segment_waveforms(clean_signals, r_peaks, fs)
            save_segments(lead_segments, save_dir, f'{prefix}_{idx}')
            save_pkl_data(save_dir, f'{prefix}_{idx}_label.pkl', labels[idx]) # labels[idx].shape = (seq_len,)
            seg_data_num +=1
            
        logger.info(f'{prefix} seg data num : {seg_data_num}')
        
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def ECG_Segmentation(downstrem_tasks, dir):
    
    fs = 360
    
    for downstrem_task in downstrem_tasks:
        processed_data_dir = os.path.join(dir, f'{downstrem_task}/ECG_Preprocessing')
        save_dir = os.path.join(dir, f'{downstrem_task}/ECG_Segmentation')
        
        process_and_save_segments(processed_data_dir, save_dir, downstrem_task, fs)