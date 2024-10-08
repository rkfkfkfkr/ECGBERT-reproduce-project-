import os
import pandas as pd
import h5py
from wfdb import rdrecord
import logging

from ECGgetdata import ECGDataProcessor
from ECGsigpreprocessing import ECGPreprocessor
from ECGsegmentation import ECGSegmentationProcessor
from ECGClustering import ECGClusterProcessor
from ECGSentenceGenerator import ECGSentenceGenerator

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# ECG Preprocessing process
class ECGPreprocessing:
    def __init__(self, 
                 org_dir: str, 
                 preprocessing_dir: str,
                 raw_data_hdf5_exists: bool,
                 clustering_model_exists: bool,
                 use_pca: bool,
                 clustering_sample_fraction: float,
                 sentnece_sample_fraction: float,
                 mask_ratio = 0.15,
                 ):
        
        self.org_dir = org_dir
        self.preprocessing_dir = preprocessing_dir
        self.raw_data_hdf5_exists = raw_data_hdf5_exists
        self.clustering_model_exists = clustering_model_exists
        self.use_pca = use_pca
        self.clustering_sample_fraction = clustering_sample_fraction
        self.sentnece_sample_fraction = sentnece_sample_fraction
        self.mask_ratio = mask_ratio
        
        self.raw_data_hdf5 = os.path.join(self.preprocessing_dir, "ecg_raw_data.hdf5")
        self.cluster_model_dir = os.path.join(self.preprocessing_dir, f"clustering_models/{clustering_sample_fraction}")
        
    def preprocessing(self):
        
        if self.raw_data_hdf5_exists == False:
            # Get raw data and saved to "ecg_raw_data.hdf5"
            raw_data_processor = ECGDataProcessor(self.org_dir)
            raw_data_processor.process_files(self.raw_data_hdf5)
            logger.info(f"Get ECG data and saved to {self.raw_data_hdf5}")
        else:
            logger.info(f"Already saved to {self.raw_data_hdf5}")
        
        # Preprocessing
        signal_preprocessing_processor = ECGPreprocessor(self.raw_data_hdf5)
        ecg_data = signal_preprocessing_processor.preprocess()
        logger.info(f"ECG signal prerpocessing Done")
        
        # Segmentation
        segments_signal_processor = ECGSegmentationProcessor(ecg_data)
        ecg_data = segments_signal_processor.preprocess()
        logger.info(f"ECG signal Segmentation Done")
        """ ecg_data = 
            {
                group_key: {
                    "signal": np.ndarray,  # (leads, seq_len)
                    "seq_len": int,
                    "fs": int,
                    "source": str
                    "segments : [ { P: np.array([[st, end], ... ]) , QRS: np.array([[st, end], ... ]), T: np.array([[st, end], ... ]), 
                    BG: np.array([[st, end], ... ]), Total: np.array([[st, end, wave_type], ... ]) } ] # (leads,)"
                    }
            }"""
        
        # Clustering
        clustering_processor = ECGClusterProcessor(self.cluster_model_dir, ecg_data, 
                                                   self.clustering_sample_fraction, 
                                                   model_exists=self.clustering_model_exists,
                                                   use_pca=self.use_pca
                                                   )
        clustering_processor.run_clustering()
        logger.info(f"ECG signal Clustering Done")
        
        # Generate Sentence
        sentence_generator = ECGSentenceGenerator(ecg_data, self.cluster_model_dir, sample_fraction=self.sentnece_sample_fraction, mask_ratio=self.mask_ratio)
        ecg_sentnece = sentence_generator.preprocess_and_return()
        logger.info(f"ECG sentence generation Done")
        """ ecg_sentence = { group_key : {sentence_token, masked_sentence_token, masked_sentence_attention_mask}}"""
        
        return ecg_sentnece 