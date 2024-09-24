import h5py
import numpy as np
import os
import joblib
import logging
import random
import json
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from transformers import BertTokenizer

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def get_tokenizer():
    custom_vocab = ['[PAD]', '[CLS]', '[SEP]', '[MASK]',  
                    'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 
                    'qrs0', 'qrs1', 'qrs2', 'qrs3', 'qrs4', 'qrs5', 'qrs6', 'qrs7', 'qrs8', 'qrs9', 
                    'qrs10', 'qrs11', 'qrs12', 'qrs13', 'qrs14', 'qrs15', 'qrs16', 'qrs17', 'qrs18', 
                    't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 
                    'bg0', 'bg1', 'bg2', 'bg3', 'bg4', 'bg5', 'bg6', 'bg7', 'bg8', 'bg9', 'bg10', 'bg11', 
                    'bg12', 'bg13', 'bg14', 'bg15', 'bg16', 'bg17', 'bg18', 'bg19', 'bg20', 'bg21', 'bg22', 
                    'bg23', 'bg24']

    vocab_file_path = "custom_vocab.txt"
    with open(vocab_file_path, 'w') as f:
        for token in custom_vocab:
            f.write(token + '\n')

    tokenizer = BertTokenizer(vocab_file=vocab_file_path)
    return tokenizer

class ECGDataHandler:
    def __init__(self, hdf5_file, n_components, mask_ratio):
        self.hdf5_file = hdf5_file
        self.n_components = n_components
        self.mask_ratio = mask_ratio
        self.tokenizer = get_tokenizer()
        self.seq_max_length = self._get_seq_max_length_()

    def _load_segments(self, group):
        segments_json = group['segments'][()].decode('utf-8')
        return json.loads(segments_json)

    def _get_seq_max_length_(self):
        max_seq_length = 0
        with h5py.File(self.hdf5_file, 'r') as hdf:
            for group_key in hdf.keys():
                signal_length = hdf[group_key]['signal'].shape[1]
                max_seq_length = max(signal_length, max_seq_length)
        return max_seq_length

    def list_to_sentence(self, arr):
        return " ".join(arr)

    def pad_wave_segments(self, wave_segments):
        max_length = max(len(ws) for ws in wave_segments)
        return np.array([np.pad(ws, (0, max_length - len(ws)), 'constant') for ws in wave_segments], dtype=np.float32)

    def extract_combined_segments(self, group, models):
        segments = self._load_segments(group)
        signal_data = group['signal'][:]
        combined_segments = []

        def process_lead(lead):
            lead_signal = signal_data[lead]
            lead_segments_total = np.array(segments[lead]["Total"])
            for idx, (st, end, wave_type) in enumerate(lead_segments_total):
                if int(st) == int(end):
                    continue
                wave_segments = lead_signal[int(st):int(end)].reshape(1, -1)
                wave_segments = np.expand_dims(wave_segments, axis=2)
                cluster = models[wave_type].predict(wave_segments)
                lead_segments_total[idx][2] = f'{wave_type}{cluster[0]}'
            return lead_segments_total

        combined_segments = Parallel(n_jobs=-1)(delayed(process_lead)(lead) for lead in range(len(signal_data)))
        return combined_segments

    def generate_masked_sentence(self, group, combined_segments):
        masked_signal = group['signal'][:]
        sentence, masked_sentence = [], []

        for lead_idx, lead_combined in enumerate(combined_segments):
            lead_masked_sentence = lead_combined[:, 2].copy()
            sentence.append(self.list_to_sentence(lead_masked_sentence.copy()))

            mask_vocab_idx = random.sample(range(len(lead_combined)), int(len(lead_combined) * self.mask_ratio))
            for idx in mask_vocab_idx:
                st, end = int(lead_combined[idx][0]), int(lead_combined[idx][1])
                masked_signal[lead_idx][st:end] = 0
                lead_masked_sentence[idx] = '[MASK]'

            masked_sentence.append(self.list_to_sentence(lead_masked_sentence))

        # Batch tokenization to handle sentences in parallel
        tokenized_sentences = self.tokenizer.batch_encode_plus(sentence, padding='max_length', max_length=self.seq_max_length, truncation=True, return_tensors='pt')
        tokenized_masked_sentences = self.tokenizer.batch_encode_plus(masked_sentence, padding='max_length', max_length=self.seq_max_length, truncation=True, return_tensors='pt')

        return (masked_signal, tokenized_sentences['input_ids'], tokenized_sentences['attention_mask'],
                tokenized_masked_sentences['input_ids'], tokenized_masked_sentences['attention_mask'])

class ECGSentenceGenerator:
    def __init__(self, hdf5_file, model_dir, output_file, sample_fraction=1.0, n_components=10, mask_ratio=0.15):
        self.data_handler = ECGDataHandler(hdf5_file, n_components, mask_ratio)
        self.model_dir = model_dir
        self.output_file = output_file
        self.models = self.load_models()
        self.sample_fraction = sample_fraction

    def load_models(self):
        models = {}
        for wave_type in ["P", "QRS", "T", "BG"]:
            model_path = os.path.join(self.model_dir, f'{wave_type}_cluster.pkl')
            models[wave_type] = joblib.load(model_path)
        return models

    def generate_sentence_for_signal(self, group):
        combined_segments = self.data_handler.extract_combined_segments(group, self.models)
        return self.data_handler.generate_masked_sentence(group, combined_segments)

    def save_results_to_hdf5(self, input_hdf5, output_hdf5):
        with h5py.File(input_hdf5, 'r') as hdf_in, h5py.File(output_hdf5, 'w') as hdf_out:
            
            # sample_keys = random.sample(list(hdf_in.keys()), int(len(hdf_in) * self.sample_fraction))
            sample_keys = list(hdf_in.keys())
            if self.sample_fraction < 1.0:
                sample_size = int(len(list(hdf_in.keys())) * self.sample_fraction)
                sample_keys = random.sample(list(hdf_in.keys()), sample_size)
            
            for group_key in tqdm(sample_keys, desc="Generate ECG Sentence", unit="group"):
                group = hdf_in[group_key]
                masked_signal, sentence_token, sentence_attention_mask, masked_sentence_token, masked_sentence_attention_mask = self.generate_sentence_for_signal(group)

                new_group = hdf_out.create_group(group_key)
                new_group.create_dataset('signal', data=group['signal'][:])
                new_group.attrs['seq_len'] = group.attrs['seq_len']
                new_group.attrs['fs'] = group.attrs['fs']
                new_group.attrs['Source'] = group.attrs['Source']

                new_group.create_dataset('masked_signal', data=masked_signal)
                new_group.create_dataset('sentence_token', data=sentence_token)
                new_group.create_dataset('sentence_attention_mask', data=sentence_attention_mask)
                new_group.create_dataset('masked_sentence_token', data=masked_sentence_token)
                new_group.create_dataset('masked_sentence_attention_mask', data=masked_sentence_attention_mask)
'''
if __name__ == "__main__":
    
    dir = "D:/data/ECGBERT/for_git4/preprocessing/"
    cluster_sample_fraction = 0.001
    sentence_sample_fraction = 0.2
    hdf5_file = os.path.join(dir, "segments_ecg_data.hdf5") 
    model_dir = os.path.join(dir, f"clustering_models/{cluster_sample_fraction}_sample/")
    output_hdf5 = os.path.join(dir, f"{sentence_sample_fraction}_sentence_ecg_data.hdf5")

    sentence_generator = ECGSentenceGenerator(hdf5_file, model_dir, output_hdf5, sample_fraction=sentence_sample_fraction, n_components=10, mask_ratio=0.15)
    sentence_generator.save_results_to_hdf5(hdf5_file, output_hdf5)
    logger.info("ECG sentence generation and saving complete.")
'''
def ECGSentenceGenerate(base_dir, cluster_sample_fraction, sentence_sample_fraction=0.2):
    
    hdf5_file = os.path.join(base_dir, "segments_ecg_data.hdf5") 
    model_dir = os.path.join(base_dir, f"clustering_models/{cluster_sample_fraction}_sample/")
    output_hdf5 = os.path.join(base_dir, f"{sentence_sample_fraction}_sentence_ecg_data.hdf5")

    sentence_generator = ECGSentenceGenerator(hdf5_file, model_dir, output_hdf5, sample_fraction=sentence_sample_fraction, n_components=10, mask_ratio=0.15)
    sentence_generator.save_results_to_hdf5(hdf5_file, output_hdf5)
    logger.info(f"ECG sentence generation and saved to {output_hdf5}")