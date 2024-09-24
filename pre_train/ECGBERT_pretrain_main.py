import os
import logging

from ECGgetdata import ECGGetData
from ECGpreprocessing import ECGPreprocessing
from ECGsegmentation import ECGSegmentation
from ECGClustering import ECGClustering
from ECGSentenceGenerator import ECGSentenceGenerate

from ECGBERT_pretrain_engine import ECGPreTrain

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    
    org_dir = " "
    preprocessing_dir = " "
    output_dir = " "
    
    # Parameter
    clustering_sample_fraction = 0.01 # default 0.01
    sentence_sample_fraction = 0.2 # default 0.2
    pretrain_sample_fraction = 0.001 # default 0.001
    
    # Preprocessing
    ECGGetData(org_dir, preprocessing_dir)
    ECGPreprocessing(preprocessing_dir)
    ECGSegmentation(preprocessing_dir)
    ECGClustering(preprocessing_dir, clustering_sample_fraction)
    ECGSentenceGenerate(preprocessing_dir, clustering_sample_fraction, sentence_sample_fraction)
    
    # Pre_train
    ECGPreTrain(preprocessing_dir, output_dir, sentence_sample_fraction, pretrain_sample_fraction)