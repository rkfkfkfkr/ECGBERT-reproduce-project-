import h5py
import random
import torch
from torch.utils.data import Dataset
import logging
import os

from ECGprerpocessing import ECGPreprocessing

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# ECGDataset 클래스
class ECGDataset(Dataset):
    def __init__(self, params):
        
        ecg_preprocessor = ECGPreprocessing( org_dir = params['dataset']['org_dir'], 
                                            preprocessing_dir = params['dataset']['preprocessing_dir'],
                                            raw_data_hdf5_exists=params['preproc']['raw_data_hdf5_exists'],
                                            clustering_model_exists = params['preproc']['clustering_model_exists'],
                                            use_pca=params['preproc']['use_pca'],
                                            clustering_sample_fraction = params['preproc']['clustering_sample_fraction'],
                                            sentnece_sample_fraction = params['preproc']['sentence_sample_fraction'],
                                            mask_ratio = params['preproc']['mask_ratio'])

        
        self.ecg_sentence = ecg_preprocessor.preprocessing()
        self.raw_data_hdf5 = os.path.join(params['dataset']['preprocessing_dir'], "ecg_raw_data.hdf5")
        self.group_keys = list(self.ecg_sentence.keys())  # 그룹 키 리스트
        self.max_seq_length = self._get_seq_length()

        pretrain_sample_fraction = params['train']['pretrain_sample_fraction']
        if pretrain_sample_fraction < 1.0:
            sample_size = max(int(len(self.group_keys) * pretrain_sample_fraction), 1)
            self.group_keys = random.sample(self.group_keys, sample_size)
        logger.info(f"ECG sentence keys_list:{len(self.group_keys)}")

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        group_key = self.group_keys[idx]
        
        with h5py.File(self.raw_data_hdf5, 'r') as hdf:
            signal = hdf[group_key]['signal'][:]  # (leads, seq_len)

        #sentence_token = torch.tensor(self.ecg_sentence[group_key]['sentence_token'], dtype=torch.long)
        #masked_sentence_token = torch.tensor(self.ecg_sentence[group_key]['masked_sentence_token'], dtype=torch.long)
        #masked_sentence_attention_mask = torch.tensor(self.ecg_sentence[group_key]['masked_sentence_attention_mask'], dtype=torch.long)
        
        sentence_token = self.ecg_sentence[group_key]['sentence_token']
        if not torch.is_tensor(sentence_token):
            sentence_token = torch.tensor(sentence_token, dtype=torch.long)
        masked_sentence_token = self.ecg_sentence[group_key]['masked_sentence_token']
        if not torch.is_tensor(masked_sentence_token):
            masked_sentence_token = torch.tensor(masked_sentence_token, dtype=torch.long)
        masked_sentence_attention_mask = self.ecg_sentence[group_key]['masked_sentence_attention_mask']
        if not torch.is_tensor(masked_sentence_attention_mask):
            masked_sentence_attention_mask = torch.tensor(masked_sentence_attention_mask, dtype=torch.long)


        return signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask

    def _get_seq_length(self, sec=10):
        # Sampling frequency for the signals
        hdf = h5py.File(self.raw_data_hdf5, 'r')
        return hdf[self.group_keys[0]].attrs['fs']*sec
    
# HDF5 파일로부터 데이터를 로드하는 ECGDataset 클래스
class LoadedECGDataset(Dataset):
    def __init__(self, hdf5_file, raw_hdf5_file):
        """
        HDF5 파일로부터 데이터를 로드하여 Dataset을 생성하는 클래스.

        Args:
            hdf5_file (str): 불러올 HDF5 파일 경로.
        """
        self.hdf5_file = hdf5_file
        self.raw_hdf5_file = raw_hdf5_file

        # HDF5 파일을 열어 group key 리스트 가져오기
        with h5py.File(self.hdf5_file, 'r') as hdf:
            self.group_keys = list(hdf.keys())

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 그룹 키를 가져오기
        group_key = self.group_keys[idx]

        # HDF5 파일 열기 (필요한 그룹 데이터만 읽기)
        with h5py.File(self.hdf5_file, 'r') as hdf:
            group = hdf[group_key]

            # 각 데이터셋을 불러오기
            signal = torch.tensor(group['signal'][:], dtype=torch.float32)
            sentence_token = torch.tensor(group['sentence_token'][:], dtype=torch.long)
            masked_sentence_token = torch.tensor(group['masked_sentence_token'][:], dtype=torch.long)
            masked_sentence_attention_mask = torch.tensor(group['masked_sentence_attention_mask'][:], dtype=torch.long)

        return signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask
    
    def _get_seq_length(self, sec=10):
        # Sampling frequency for the signals
        hdf = h5py.File(self.raw_hdf5_file, 'r')
        return hdf[self.group_keys[0]].attrs['fs']*sec
