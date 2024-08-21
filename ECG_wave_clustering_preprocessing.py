import numpy as np
import os
import pickle
from tslearn.clustering import TimeSeriesKMeans
import time
import torch
from models import KMeansClusteringGPU

def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pkl_data(save_dir, file_name, save_pkl):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_pkl, f)
    print('pkl file saved')
    
def reshape_signals(wave_type_signals, bg_max_length=0):
    # Calculate the number of signals and the maximum length
    max_length = max(len(signal) for signal in wave_type_signals)

    if bg_max_length > 0:
        max_length = bg_max_length

    # Convert each signal to a PyTorch tensor, pad them, and store them in a list
    padded_signals = [
        torch.nn.functional.pad(torch.tensor(signal, device='cuda'), (0, max_length - len(signal)))
        for signal in wave_type_signals
    ]

    # Stack the padded signals into a single tensor and move to the GPU
    reshaped_signals = torch.stack(padded_signals).to('cuda')
    
    return reshaped_signals

def kmeans_clustering(X, n_clusters, save_dir, wave_type, batch_num):

    start_t = time.time()
    kmeans = KMeansClusteringGPU(n_clusters)

    if batch_num == 1:
        kmeans.fit(X)
    else:
        batch_size = 1000#int(len(X)/batch_num)
        max_length = max(len(signal) for signal in X)
        for i in range(0, len(X), batch_size):
            print(f'partial_fit {int(i/batch_size)}/{int(len(X)/batch_size)+1}')
            reshape_X = reshape_signals(X[i:i+batch_size], bg_max_length=max_length)
            kmeans.partial_fit(reshape_X)

    cluster_t= time.time() - start_t
    print(f'{wave_type}-clustering _ t: {(cluster_t):.3f}')
    
    save_pkl_data(save_dir, f'{wave_type}_cluster_centorids.pkl', kmeans.centroids)
    
def process_wave_type_segment(prefix_seg_data, preprocessed_signal):
    prefix_preprocessed_data = []
    for lead in range(len(prefix_seg_data)):
        for (st, end) in prefix_seg_data[lead][0]:
            prefix_preprocessed_data.append(preprocessed_signal[:,lead][st:(end+1)])
    return prefix_preprocessed_data

def clustering_and_save_wave(seg_dir, processed_data_dir, save_dir):
    wave_types_clusters = {'p': (12,1), 'qrs': (19,1), 't': (14,1), 'bg': (25,2)}
    prefix_file_num = {
        'cpsc_train': 5501, 'cpsc_val': 1376, 'georgia_train': 8270,
        'georgia_val': 2068, 'ptb_xl_train': 17439, 'ptb_xl_val': 4360
    }
    for wave_type, (n_clusters, batch_num) in wave_types_clusters.items():

        # save wave_type_signals
        wave_type_signals = []
        for prefix, prefix_data_num in prefix_file_num.items():
            preprocessed_signals = load_pkl_data(os.path.join(processed_data_dir, f'{prefix}_processed_signals.pkl'))
            for prefix_data_idx in range(prefix_data_num):
                preprocessed_signal = preprocessed_signals[prefix_data_idx]
                prefix_data_file = f'{prefix}_{prefix_data_idx}_{wave_type}_segments.pkl'
                prefix_seg_data = load_pkl_data(os.path.join(seg_dir, os.path.join(wave_type, prefix_data_file)))
                prefix_preprocessed_data = process_wave_type_segment(prefix_seg_data, preprocessed_signal)
                wave_type_signals.extend(prefix_preprocessed_data)
            print(f'{wave_type}-{prefix}')
            
        save_pkl_data(save_dir, f'{wave_type}_signals.pkl', wave_type_signals)

        if wave_type != 'bg':
            file_path = os.path.join(save_dir, f'{wave_type}_signals.pkl')
            wave_type_signals = load_pkl_data(file_path)

            # savereshaped_signals
            reshaped_signals = reshape_signals(wave_type_signals)
            wave_type_signals.clear()
            save_pkl_data(save_dir, f'{wave_type}_cluster_X.pkl', reshaped_signals)

        print(f'{wave_type} clustering_preprocessing Done')

        
if __name__ == '__main__':
    processed_data_dir = 'D:/data/ECGBERT/for_git3/preprocessing/ECG_preprocessing/'
    seg_dir = 'D:/data/ECGBERT/for_git3/preprocessing/ECG_segmentation/'
    save_dir = 'D:/data/ECGBERT/for_git3/preprocessing/ECG_Clustering/'
    
    clustering_and_save_wave(seg_dir, processed_data_dir, save_dir)
