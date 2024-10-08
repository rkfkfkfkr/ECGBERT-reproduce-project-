import numpy as np
import random
import os
import joblib
from joblib import Parallel, delayed
from transformers import BertTokenizer
from tqdm import tqdm

# Custom logger setup
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

# Tokenizer 설정 함수
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

# 데이터 핸들러 클래스
class ECGDataHandler:
    def __init__(self, ecg_data_dict, n_components, mask_ratio):
        self.ecg_data_dict = ecg_data_dict
        self.n_components = n_components
        self.mask_ratio = mask_ratio
        self.tokenizer = get_tokenizer()
        self.seq_max_length = self._get_seq_length()
        logger.info(f"seq_max_length: {self.seq_max_length}")

    def list_to_sentence(self, arr):
        return " ".join(arr)
    
    def extract_combined_segments(self, group_data, models):
        segments = group_data["segments"]
        signal_data = group_data['signal']
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
    
    def _get_seq_length(self):
        group_key = list(self.ecg_data_dict.keys())[0]
        group_data = self.ecg_data_dict[group_key]
        return group_data['signal'].shape[1]
        
    def generate_masked_sentence(self, combined_segments):
        """생성된 segment에서 마스킹된 문장을 생성."""
        sentence, masked_sentence = [], []

        for lead_combined in combined_segments:
            lead_masked_sentence = lead_combined[:, 2].copy()
            sentence.append(self.list_to_sentence(lead_masked_sentence.copy()))

            mask_vocab_idx = random.sample(range(len(lead_combined)), int(len(lead_combined) * self.mask_ratio))
            for idx in mask_vocab_idx:
                lead_masked_sentence[idx] = '[MASK]'

            masked_sentence.append(self.list_to_sentence(lead_masked_sentence))

        # Batch tokenization for sentences
        tokenized_sentences = self.tokenizer.batch_encode_plus(sentence, padding='max_length', max_length=self.seq_max_length, truncation=True, return_tensors='pt')
        tokenized_masked_sentences = self.tokenizer.batch_encode_plus(masked_sentence, padding='max_length', max_length=self.seq_max_length, truncation=True, return_tensors='pt')

        return (tokenized_sentences['input_ids'], tokenized_masked_sentences['input_ids'], tokenized_masked_sentences['attention_mask'])

class ECGSentenceGenerator:
    def __init__(self, ecg_data_dict, model_dir, sample_fraction, mask_ratio, n_components=10):
        self.data_handler = ECGDataHandler(ecg_data_dict, n_components, mask_ratio)
        self.models = self.load_models(model_dir)
        self.sample_fraction = sample_fraction

    def load_models(self, model_dir):
        """Load clustering models for each wave type."""
        models = {}
        for wave_type in ["P", "QRS", "T", "BG"]:
            model_path = os.path.join(model_dir, f'{wave_type}_cluster.pkl')
            models[wave_type] = joblib.load(model_path)
        return models

    def generate_sentence_for_signal(self, group_key):
        """입력 데이터에서 문장과 마스크 문장을 생성."""
        group_data = self.data_handler.ecg_data_dict[group_key]
        combined_segments = self.data_handler.extract_combined_segments(group_data, self.models)
        return self.data_handler.generate_masked_sentence(combined_segments)

    def preprocess_and_return(self):
        """ecg_data_dict에서 각 그룹에 대해 문장을 생성하고 dict로 반환."""
        result_dict = {}

        sample_keys = list(self.data_handler.ecg_data_dict.keys())
        if self.sample_fraction < 1.0:
            sample_size = int(len(list(self.data_handler.ecg_data_dict.keys())) * self.sample_fraction)
            sample_keys = random.sample(list(self.data_handler.ecg_data_dict.keys()), sample_size)
        logger.info(f"ECG sentence sample fraction: {self.sample_fraction}")
        logger.info(f"Number of sample_keys: {len(sample_keys)}")
        logger.info(f"Sample keys: {sample_keys[:10]}")

        for group_key in tqdm(sample_keys, desc="Generate ECG Sentence", unit="group"):
            # 문장 생성 및 결합
            sentence_token, masked_sentence_token, masked_sentence_attention_mask = self.generate_sentence_for_signal(group_key)

            result_dict[group_key] = {
                "sentence_token": sentence_token,
                "masked_sentence_token": masked_sentence_token,
                "masked_sentence_attention_mask": masked_sentence_attention_mask
            }
            
            #
            if group_key == sample_keys[0]:
                logger.info(group_key)
                logger.info(f"sentence_token : f{result_dict[group_key]['sentence_token'].shape}")
                logger.info(f"masked_sentence_token : f{result_dict[group_key]['masked_sentence_token'].shape}")
                logger.info(f"masked_sentence_attention_mask : f{result_dict[group_key]['masked_sentence_attention_mask'].shape}")
    
        logger.info(f"ECG sentence result_dict: {len(result_dict)}")
        return result_dict