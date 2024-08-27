import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from itertools import groupby
from joblib import Parallel, delayed

def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pkl_data(save_dir, file_name, save_pkl):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_pkl, f)

def process_wave_type_segment(lead_signal_segments, preprocessed_signal_lead):
    segments = [preprocessed_signal_lead[st:end+1] for (st, end) in lead_signal_segments]
    return segments

def calculate_distance(signal, cluster_centers):
    signal_len = len(signal)
    cluster_len = cluster_centers.shape[1]
    
    if signal_len != cluster_len:
        signal = np.pad(signal, (0, max(cluster_len - signal_len, 0)), 'constant')[:cluster_len]
    
    distances = euclidean_distances([signal], cluster_centers)
    return distances

def process_lead_signal(lead, signal_segments, preprocessed_signal, cluster_dir, wave_types):
    lead_signal_vocab = np.zeros(preprocessed_signal.shape[0])
    
    for wave_type in wave_types:
        wave_type_preprocessed_signal = process_wave_type_segment(signal_segments[lead][wave_type][0], preprocessed_signal[:, lead])

        cluster_centers = load_pkl_data(os.path.join(cluster_dir, f'{wave_type}_cluster_centorids.pkl')).cpu().numpy()

        for seg_idx, signal in enumerate(wave_type_preprocessed_signal):
            distances = calculate_distance(signal, cluster_centers)
            cluster_idx = np.argmin(distances) + {'p': 0, 'qrs': 12, 't': 31, 'bg': 45}[wave_type]

            idx_st, idx_end = signal_segments[lead][wave_type][0][seg_idx]
            lead_signal_vocab[idx_st:idx_end+1] = int(cluster_idx)
    
    return lead_signal_vocab

def vocab_create_assignment(downstrem_task, processed_data_dir, seg_dir, cluster_dir, save_dir):
    wave_types = ['p', 'qrs', 't', 'bg']
    
    for suffix in ['train', 'val']:
        t_sentence_num = 0
        v_sentence_num = 0
        
        prefix = f'{downstrem_task}_{suffix}'
        preprocessed_signals = load_pkl_data(os.path.join(processed_data_dir, f'{prefix}_processed_signals.pkl'))
                
        for idx, preprocessed_signal in enumerate(preprocessed_signals):
            signal_segments = load_pkl_data(os.path.join(seg_dir, f'{prefix}_{idx}_segments.pkl'))
            labels = load_pkl_data(os.path.join(seg_dir, f'{prefix}_{idx}_label.pkl'))

            signal_vocabs = Parallel(n_jobs=-1)(
                delayed(process_lead_signal)(lead, signal_segments, preprocessed_signal, cluster_dir, wave_types) 
                for lead in range(len(signal_segments))
            )

            for lead, lead_signal_vocab in enumerate(signal_vocabs):
                sentence_signal = preprocessed_signal[:, lead]
                sentence = [71] + lead_signal_vocab.tolist() + [72]  # Convert numpy array to list
                sentence_label = labels 
            
                data = [sentence, sentence_signal, sentence_label]
                if prefix.endswith('train'):
                    save_pkl_data(os.path.join(save_dir, 'train'), f'sentence_{t_sentence_num}.pkl', data)
                    t_sentence_num += 1
                else:
                    save_pkl_data(os.path.join(save_dir, 'val'), f'sentence_{v_sentence_num}.pkl', data)
                    v_sentence_num += 1

        logger.info(f'{prefix} Sentence Done')

import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def ECG_Beat_Sentence(downstrem_tasks, dir, cluster_dir):
    
    for downstrem_task in downstrem_tasks:
        processed_data_dir = os.path.join(dir, f'{downstrem_task}/ECG_Preprocessing')
        seg_dir = os.path.join(dir, f'{downstrem_task}/ECG_Segmentation')
        save_dir = os.path.join(dir, f'{downstrem_task}/ECG_Sentence')
        
        vocab_create_assignment(downstrem_task, processed_data_dir, seg_dir, cluster_dir, save_dir)
