import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from itertools import groupby
import random

def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pkl_data(save_dir, file_name, save_pkl):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_pkl, f)
    #print(f'pkl file saved to {save_path}')

def process_wave_type_segment(lead_signal_segments, preprocessed_signal_lead):
    # 벡터화 연산을 통해 빠르게 처리
    segments = [preprocessed_signal_lead[st:end+1] for (st, end) in lead_signal_segments]
    return segments

def calculate_distance(signal, cluster_centers):
    signal_len = len(signal)
    cluster_len = cluster_centers.shape[1]
    
    if signal_len != cluster_len:
        signal = np.pad(signal, (0, max(cluster_len - signal_len, 0)), 'constant')[:cluster_len]
    
    distances = euclidean_distances([signal], cluster_centers)
    return distances

def generate_sentence_and_masking(lead_sig, preprocessed_signal, lead):
    
    def compress_signal(signal):
        return [key for key, group in groupby(signal)], [len(list(group)) for key, group in groupby(signal)]

    def find_beats(lead_sig):
        beats = []
        current_beat = []
        beat_start = 0

        for i, value in enumerate(lead_sig):
            if int(value) < 12:
                if current_beat and len(current_beat) > 2:
                    beats.append((current_beat, beat_start, i - 1))
                current_beat = [int(value)]
                beat_start = i
            else:
                current_beat.append(int(value))
            
        if current_beat and len(current_beat) > 3:
            beats.append((current_beat, beat_start, len(lead_sig) - 1))
            
        return beats

    def extract_random_beats(beats):
        num_beats = random.choice([1, 2, 6, 8])
        num_beats = min(num_beats, len(beats))
        
        start_idx = random.randint(0, len(beats) - num_beats)
        selected_beats = beats[start_idx:start_idx + num_beats]
        sentence = [beat for beat, _, _ in selected_beats]
        st_end_indices = [selected_beats[0][1], selected_beats[-1][2]]
            
        return np.concatenate(sentence), st_end_indices

    comp_sig, comp_sig_len = compress_signal(lead_sig)
    beats = find_beats(comp_sig)
    if len(beats) < 1:
        return [], []
    sentence_comp, st_end_idx_comp = extract_random_beats(beats)

    choice_num = int(len(sentence_comp) * 0.15)
    if choice_num < 1:
        choice_num = 1
    Masking_comp_idx = np.random.choice(range(0, len(sentence_comp)-1),  choice_num)
    
    st_idx, end_idx = st_end_idx_comp
    sentence = [71]  # CLS tokens
    sentence.extend(np.repeat(sentence_comp, comp_sig_len[st_idx:end_idx+1]))
    sentence.append(72)  # SEP token
    
    sentence_st_end_idx = [sum(comp_sig_len[:st_idx]), sum(comp_sig_len[:st_idx])+len(sentence)-1]
    sentence_signal = preprocessed_signal[:,lead][sentence_st_end_idx[0]:sentence_st_end_idx[1]+1]
    
    sentence_comp[Masking_comp_idx] = 73
    Masked_sentence = [71]  # CLS token
    Masked_sentence.extend(np.repeat(sentence_comp, comp_sig_len[st_idx:end_idx+1]))
    Masked_sentence.append(72)  # SEP token

    Masked_sentence_st_end_idx = [sum(comp_sig_len[:st_idx]), sum(comp_sig_len[:st_idx])+len(Masked_sentence)-1]
    Masked_signal = preprocessed_signal[:,lead][Masked_sentence_st_end_idx[0]:Masked_sentence_st_end_idx[1]+1]

    return sentence, sentence_signal, Masked_sentence, Masked_signal


def vocab_create_assignment(processed_data_dir, seg_dir, cluster_dir, save_dir, batch_size=1024):
    wave_types = ['p', 'qrs', 't', 'bg']
    prefixes = ['cpsc_train', 'cpsc_val', 'georgia_train', 'georgia_val', 'ptb_xl_train', 'ptb_xl_val']

    t_sentence_num = 0
    v_sentence_num = 0
    batch_data = []

    for prefix in prefixes:
        preprocessed_signals = load_pkl_data(os.path.join(processed_data_dir, f'{prefix}_processed_signals.pkl'))
            
        for idx, preprocessed_signal in enumerate(preprocessed_signals):
            preprocessed_signal = preprocessed_signals[idx]
            signal_segments = load_pkl_data(os.path.join(seg_dir, f'{prefix}_{idx}_segments.pkl'))

            for lead in range(len(signal_segments)):
                lead_signal_vocab = np.zeros(preprocessed_signal.shape[0])
                
                for wave_type in wave_types:
                    wave_type_preprocessed_signal = process_wave_type_segment(signal_segments[lead][wave_type][0], preprocessed_signal[:, lead])

                    cluster_centers = load_pkl_data(os.path.join(cluster_dir, f'{wave_type}_cluster_centorids.pkl')).cpu().numpy()

                    for seg_idx, signal in enumerate(wave_type_preprocessed_signal):
                        distances = calculate_distance(signal, cluster_centers)
                        cluster_idx = np.argmin(distances) + {'p': 0, 'qrs': 12, 't': 31, 'bg': 45}[wave_type]
                        
                        idx_st, idx_end = signal_segments[lead][wave_type][0][seg_idx]
                        lead_signal_vocab[idx_st:idx_end+1] = int(cluster_idx)

                sentence, sentence_signal, Masked_sentence, Masked_signal  = generate_sentence_and_masking(lead_signal_vocab, preprocessed_signal, lead)
                if len(Masked_sentence) > 0:
                    sentence_save_data = [sentence, sentence_signal, Masked_sentence, Masked_signal]
                    batch_data.append(sentence_save_data)

                if len(batch_data) >= batch_size:
                    for data in batch_data:
                        if prefix.endswith('train'):
                            save_pkl_data(os.path.join(save_dir, 'train'), f'sentence_{t_sentence_num}.pkl', data)
                            t_sentence_num += 1
                        else:
                            save_pkl_data(os.path.join(save_dir, 'val'), f'sentence_{v_sentence_num}.pkl', data)
                            v_sentence_num += 1
                    batch_data.clear()
                    
        if batch_data:
            for data in batch_data:
                if prefix.endswith('train'):
                    save_pkl_data(os.path.join(save_dir, 'train'), f'sentence_{t_sentence_num}.pkl', data)
                    t_sentence_num += 1
                else:
                    save_pkl_data(os.path.join(save_dir, 'val'), f'sentence_{v_sentence_num}.pkl', data)
                    v_sentence_num += 1
            batch_data.clear()
        logger.info(f'{prefix} Sentence Done')

import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def ECG_Beat_Sentence(dir):
    
    processed_data_dir = os.path.join(dir, f'ECG_Preprocessing')
    seg_dir = os.path.join(dir, f'ECG_Segmentation')
    cluster_dir = os.path.join(dir, f'ECG_Clustering')
    save_dir = os.path.join(dir, f'ECG_Sentence')
    
    vocab_create_assignment(processed_data_dir, seg_dir, cluster_dir, save_dir)
    
    
    
