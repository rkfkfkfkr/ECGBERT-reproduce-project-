import os
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from utils.misc import yaml_load

from ECGDataset import ECGDataset, LoadedECGDataset
from test import BERT_pretrain

import logging
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def get_args_parser():
    
    parser = argparse.ArgumentParser('ECGBERT pre-training', add_help=False)

    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='ECGBERT', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--seq_len', default=5000, type=int,
                        help='Sequence length (number of time points) of ECG input')
    parser.add_argument('--num_leads', default=12, type=int,
                        help='Number of leads of ECG input')

    # MELM parameters : Masked ECG Langauge Model
    parser.add_argument('--mask_ratio', default=0.15, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Optimizer 
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')

    # yaml
    parser.add_argument('--cfg_file', default='C:/Users/user/hwang/ECGBERT/for_git5/pre_train.yaml', type=str,
                        help="Config file name under the config directory.")    
    parser.add_argument('--override_yaml', type=str, default=None)

    return parser

import h5py
def save_dataset_to_hdf5(dataset, output_hdf5_file):
    """
    ECGDataset의 데이터를 HDF5 파일로 저장하는 함수.
    
    Args:
        dataset (ECGDataset): 저장할 데이터셋 객체.
        output_hdf5_file (str): 저장할 HDF5 파일 경로.
    """
    # HDF5 파일 생성 및 쓰기 모드로 열기
    with h5py.File(output_hdf5_file, 'w') as hdf5_file:
        # 데이터셋 전체를 순회하며 각 그룹 데이터 저장
        for idx, (signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask) in enumerate(dataset):
            group_key = dataset.group_keys[idx]

            # 각 그룹 생성
            group = hdf5_file.create_group(group_key)

            # 각 데이터를 HDF5에 저장
            group.create_dataset('signal', data=signal, compression="gzip", compression_opts=9)
            group.create_dataset('sentence_token', data=sentence_token.numpy(), compression="gzip", compression_opts=9)
            group.create_dataset('masked_sentence_token', data=masked_sentence_token.numpy(), compression="gzip", compression_opts=9)
            group.create_dataset('masked_sentence_attention_mask', data=masked_sentence_attention_mask.numpy(), compression="gzip", compression_opts=9)
        
        logger.info(f"Dataset has been saved to {output_hdf5_file}.")

def main(args):
    
    # check params 
    params = yaml_load(args.cfg_file, args.override_yaml)
    if params['dataset']['output_dir']:
        Path(params['dataset']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(params['train']['device'])
    cudnn.benchmark = True
    
    logger.info(f'Pre_train Device: {device}')
    
    # Preprocessing & ECGDataset
    dataset_hdf5_file = os.path.join(params['dataset']['preprocessing_dir'], f"{params['preproc']['sentence_sample_fraction']}_ECGBERT_dataset.hdf5")
    if params['train']['train_data_exists'] == False:
        dataset = ECGDataset(params)
        max_seq_len = dataset._get_seq_length()
        save_dataset_to_hdf5(dataset, dataset_hdf5_file)
        logger.info("ECGPreprocessing all Done")
    else:
        raw_data_hdf5_file = os.path.join(params['dataset']['preprocessing_dir'], "ecg_raw_data.hdf5")
        dataset = LoadedECGDataset(dataset_hdf5_file, raw_data_hdf5_file)
        max_seq_len = dataset._get_seq_length()
        logger.info("Load Pre_Train ECGDataset")
    
    # pre_train
    BERT_pretrain(params, dataset, max_seq_len, device)
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)