from ECG_Preprocessing import ECG_Preprocessing
from ECG_Segmentation import ECG_Segmentation
from ECG_Clustering_Preprocessing import ECG_Clustering_Preprocessing
from ECG_Clustering import ECG_Clustering
from ECG_Beat_Sentence import ECG_Beat_Sentence
from Pre_train_engine import Pre_train_engine

import os
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    
    dir = 'D:/data/ECGBERT/pre_train/'
    org_data_dir = 'D:/data/ECGBERT/org_data/pre_train/'
    
    dataset_paths = [ ['cpsc', os.path.join(org_data_dir, 'cpsc'), '.mat'],
                      ['georgia', os.path.join(org_data_dir, 'georgia'), '.mat'],
                      ['ptb_xl', os.path.join(org_data_dir, 'ptb_xl'), '.dat']]

    datasets = [item[0] for item in dataset_paths]
    
    # Preprocessing
    ECG_Preprocessing(dataset_paths, dir)
    ECG_Segmentation(datasets, dir)
    ECG_Clustering_Preprocessing(dir)
    ECG_Clustering(dir)
    ECG_Beat_Sentence(dir)
    
    # Pre_train
    Pre_train_engine(dir)