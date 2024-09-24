import h5py
import numpy as np
import os
import random
import logging
import json
import joblib
import time
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

logging.basicConfig(level='INFO', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Class for handling ECG data loading and processing from HDF5
class ECGDataHandler:
    def __init__(self, hdf5_file, sample_fraction=1.0):
        self.hdf5_file = hdf5_file
        self.sample_fraction = sample_fraction

    def extract_wave_segments(self, wave_type):
        """Extract wave segments for the given wave type (P, QRS, T, BG) from HDF5."""
        wave_segments = []
        with h5py.File(self.hdf5_file, 'r') as hdf:
            
            sample_keys = list(hdf.keys())
            if self.sample_fraction < 1.0:
                sample_size = int(len(sample_keys) * self.sample_fraction)
                sample_keys = random.sample(sample_keys, sample_size)

            # Process each group in sampled keys
            for group_key in sample_keys:
                group = hdf[group_key]
                segment_data = self._load_segments(group)

                # Extract the signal and segment data
                signals = group['signal'][:]  # Load signals directly from HDF5

                # Vectorized extraction of lead segments
                for lead_idx in range(signals.shape[0]):
                    lead_segments = np.array(segment_data[lead_idx][wave_type])
                    # Efficient slicing and segment extraction
                    segments_signal_for_lead = [
                    signals[lead_idx][start:end] for start, end in lead_segments]
                    wave_segments.extend(segments_signal_for_lead)

        return wave_segments

    @staticmethod
    def _load_segments(group):
        """Load segments from the HDF5 file and return them as a dict."""
        segments_json = group['segments'][()].decode('utf-8')
        return json.loads(segments_json)

    @staticmethod
    def pad_wave_segments(wave_segments, n_jobs=-1):
        """Pad wave segments to equal lengths and convert to float32 for memory efficiency."""
        max_length = max(len(ws) for ws in wave_segments)

        # Preallocate the array for efficiency
        padded_segments = np.zeros((len(wave_segments), max_length), dtype=np.float32)

        # Function to pad each segment
        def pad_single_segment(ws):
            padded = np.pad(ws, (0, max_length - len(ws)), 'constant')
            return padded

        # Apply padding in parallel
        padded_segments = Parallel(n_jobs=n_jobs)(delayed(pad_single_segment)(ws) for ws in wave_segments)
        return np.array(padded_segments, dtype=np.float32)


# Class for handling clustering logic
class ECGClustering:
    def __init__(self, model_dir='clustering_models', use_pca=True, n_components=10):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.use_pca = use_pca
        self.n_components = n_components

    def cluster_waves(self, wave_segments, n_clusters, dtw_metric=True):
        """Cluster the wave segments using KMeans or DTW-KMeans."""
        logger.info(f"Clustering with {n_clusters} clusters.")
        
        # Pad wave segments
        wave_segments_padded = ECGDataHandler.pad_wave_segments(wave_segments)
        logger.info(f"Padded segments shape: {wave_segments_padded.shape}")
        
        # Apply PCA for dimensionality reduction (optional)
        if self.use_pca and self.n_components < wave_segments_padded.shape[1]:
            pca = PCA(n_components=self.n_components)
            wave_segments_padded = pca.fit_transform(wave_segments_padded)
            logger.info(f"Shape after PCA: {wave_segments_padded.shape}")
        
        # Select clustering method
        if dtw_metric:
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=False, random_state=0, n_jobs=-1)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=-1)
        
        logger.info("Fitting model ...")
        model.fit(wave_segments_padded)
        return model

    def save_model(self, model, wave_type):
        """Save the clustering model to a file."""
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        model_path = os.path.join(self.model_dir, f'{wave_type}_cluster.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Model for {wave_type} saved at {model_path}")


# Class for running the overall clustering process
class ECGClusterProcessor:
    def __init__(self, hdf5_file, sample_fraction, model_exists=False, model_dir='clustering_models'):
        self.data_handler = ECGDataHandler(hdf5_file, sample_fraction)
        self.clustering = ECGClustering(model_dir, use_pca=True)
        self.model_exists = model_exists
        self.wave_clusters = {
            "P": 12,
            "QRS": 19,
            "T": 14,
            "BG": 25
        }

    def run_clustering(self):
        """Run clustering for P, QRS, T, and BG waveforms."""
        if not self.model_exists:
            logger.info("Starting clustering with new models.")
            
            # 전체 시간 측정을 위한 시작 시간
            total_start_time = time.time()

            for wave_type, n_clusters in self.wave_clusters.items():
                # wave_type 별 시작 시간 측정
                start_time = time.time()

                logger.info(f"Processing wave type: {wave_type}")
                
                # Extract wave segments for the specific wave type
                wave_segments = self.data_handler.extract_wave_segments(wave_type)

                if len(wave_segments) == 0:
                    logger.warning(f"No wave segments found for {wave_type}, skipping.")
                    continue
                
                logger.info(f"{wave_type} sample size: {len(wave_segments)}")
                
                # Cluster the sampled wave segments
                model = self.clustering.cluster_waves(wave_segments, n_clusters, dtw_metric=True)
                
                # Save the trained model
                self.clustering.save_model(model, wave_type)

                # wave_type 처리 시간 측정 후 출력
                elapsed_time = time.time() - start_time
                logger.info(f"Time taken for {wave_type}: {int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s")
            
            # 전체 처리 시간 측정 후 출력
            total_elapsed_time = time.time() - total_start_time
            logger.info(f"Total time taken: {int(total_elapsed_time // 3600)}h {int((total_elapsed_time % 3600) // 60)}m {int(total_elapsed_time % 60)}s")
        else:
            logger.info("Models already exist. Skipping clustering.")

'''
# Usage Example
if __name__ == "__main__":
    hdf5_file = "D:/data/ECGBERT/for_git4/preprocessing/segments_ecg_data.hdf5"
    sample_fraction = 0.01
    model_dir = f'D:/data/ECGBERT/for_git4/preprocessing/clustering_models/{sample_fraction}_sample/'
    
    file_handler = logging.FileHandler(os.path.join(model_dir, 'clustering_log.txt'))
    logging.getLogger().addHandler(file_handler)
    
    # Initialize and run clustering
    processor = ECGClusterProcessor(hdf5_file, sample_fraction=sample_fraction, model_exists=False, model_dir=model_dir)
    processor.run_clustering()
    
    # parameter
    #hdf5_file, sample_fraction, model_exists, model_dir, class ECGClustering - use_pca
'''

def ECGClustering(base_dir, clustering_sample_fraction = 0.01):
    
    hdf5_file = os.path.join(base_dir, "segments_ecg_data.hdf5")
    model_dir = os.path.join(base_dir, f"clustering_models/{clustering_sample_fraction}_sample")
    file_handler = logging.FileHandler(os.path.join(model_dir, 'clustering_log.txt'))
    logging.getLogger().addHandler(file_handler)
    
    # Initialize and run clustering
    processor = ECGClusterProcessor(hdf5_file, sample_fraction=clustering_sample_fraction, model_exists=False, model_dir=model_dir)
    processor.run_clustering()
    
    logger.info(f"Clustering ECG data and saved to {model_dir}")
    
    