from ECG_Preprocessing import ECG_Preprocessing
from ECG_Segmentation import ECG_Segmentation
from ECG_Beat_Sentence import ECG_Beat_Sentence
from Fine_tune_engine import Fine_tune_engine

import os
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    
    dir = 'D:/data/ECGBERT/for_git3/fine_tune/'
    org_data_dir = 'D:/data/ECGBERT/org_data/downstream/'
    cluster_dir = 'D:/data/ECGBERT/for_git3/preprocessing/ECG_Clustering/' # cluster center save_dir
    pre_train_model_dir = 'D:/data/ECGBERT/for_git3/pre_train/results2/'
    
    dataset_paths = [ ['AFIB_Detection', os.path.join(org_data_dir, 'mit_bih_arrhythmia'), '.dat']]

    downstream_tasks = [item[0] for item in dataset_paths]
    
    # Preprocessing
    #ECG_Preprocessing(dataset_paths, dir)
    #ECG_Segmentation(downstream_tasks, dir)
    #ECG_Beat_Sentence(downstream_tasks, dir, cluster_dir)
    
    # Fine_tune
    Fine_tune_engine(downstream_tasks, pre_train_model_dir, dir)