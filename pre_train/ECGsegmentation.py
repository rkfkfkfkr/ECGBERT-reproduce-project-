import h5py
import numpy as np
import logging
import neurokit2 as nk
from joblib import Parallel, delayed
import json
from tqdm import tqdm
import os

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# Class for loading ECG data from HDF5
class ECGDataLoader:
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def load_group_data(self, group_key):
        """Load ECG data for a specific group_key (12-lead data) from HDF5 file."""
        with h5py.File(self.hdf5_file, 'r') as hdf:
            group_data = hdf[group_key]
            signals = group_data['signal'][:]  # 12-lead signal data
            seq_len = group_data.attrs['seq_len']
            fs = group_data.attrs['fs']
            source = group_data.attrs['Source']
        return signals, seq_len, fs, source

# Class for detecting R-peaks in 12-lead ECG signals
class RPeakDetector:
    @staticmethod
    def detect_r_peaks(signals, fs):
        """Detect R-peaks for all 12 leads using parallel processing."""
        
        # Clean ECG signals using parallel processing
        clean_signals = Parallel(n_jobs=-1)(delayed(nk.ecg_clean)(signals[i], sampling_rate=fs, method='hamilton2002') for i in range(signals.shape[0]))

        # Detect R-peaks for each lead using parallel processing
        all_r_peaks = Parallel(n_jobs=-1)(delayed(RPeakDetector._extract_r_peaks)(clean_signals[i], fs) for i in range(signals.shape[0]))
        
        return all_r_peaks

    @staticmethod
    def _extract_r_peaks(clean_signal, fs):
        """Helper method to extract R-peaks from a cleaned signal."""
        _, r_peak_info = nk.ecg_peaks(clean_signal, sampling_rate=fs, method='hamilton2002')
        return r_peak_info["ECG_R_Peaks"]

# Class for segmenting ECG waves (P, QRS, T, BG) in 12-lead ECG signals
class ECGWaveSegmenter:
    @staticmethod
    def segment_waves(signals, seq_len, all_r_peaks, fs):
        """Segment 12-lead signals into P, QRS, T waves and Background (BG)."""
        all_segments = []

        # signals shape = (lead, seq_len)
        for lead_idx in range(signals.shape[0]):
            lead_r_peaks = np.array(all_r_peaks[lead_idx])  # R-peaks for this lead
            signal_length = seq_len

            # Calculate segment start and end indices using vectorized operations
            qrs_start = np.maximum(lead_r_peaks - int(0.05 * fs), 0)
            qrs_end = np.minimum(lead_r_peaks + int(0.05 * fs), signal_length)
            p_start = np.maximum(qrs_start - int(0.05 * fs), 0)
            t_end = np.minimum(lead_r_peaks + int(0.15 * fs), signal_length)

            segments = {
                'P': np.column_stack([p_start, qrs_start]),
                'QRS': np.column_stack([qrs_start, qrs_end]),
                'T': np.column_stack([qrs_end, t_end]),
            }

            # Background (BG) segments
            bg_start = np.concatenate(([0], t_end))
            bg_end = np.concatenate((p_start, [signal_length]))
            bg_mask = bg_start < bg_end
            segments['BG'] = np.column_stack([bg_start[bg_mask], bg_end[bg_mask]])

            # Combine all segments into a single array with labels
            total_segments = np.concatenate([
                np.column_stack((segments['P'], np.full(len(segments['P']), 'P', dtype=object))),
                np.column_stack((segments['QRS'], np.full(len(segments['QRS']), 'QRS', dtype=object))),
                np.column_stack((segments['T'], np.full(len(segments['T']), 'T', dtype=object))),
                np.column_stack((segments['BG'], np.full(len(segments['BG']), 'BG', dtype=object))),
            ])

            # Sort the segments based on start times (first column)
            total_segments = total_segments[np.argsort(total_segments[:, 0].astype(int))]
            segments["Total"] = total_segments

            all_segments.append(segments)

        return all_segments

# Class for saving results to HDF5
class HDF5Saver:
    def __init__(self, output_file):
        self.output_file = output_file

    def open_file(self):
        """Open HDF5 file for writing."""
        self.hdf = h5py.File(self.output_file, 'w')

    def close_file(self):
        """Close the HDF5 file."""
        if self.hdf:
            self.hdf.close()

    def save_results(self, group_key, signals, segments, seq_len, fs, source):
        """Save the segmentation results and signals for each group_key."""
        group = self.hdf.create_group(group_key)
        group.create_dataset('signal', data=signals)

        # Convert segments to JSON serializable format
        segments_serializable = []
        for lead_segments in segments:
            lead_serializable = {}
            for segment_type, segment_values in lead_segments.items():
                if isinstance(segment_values, np.ndarray):
                    lead_serializable[segment_type] = segment_values.tolist()  # Convert ndarray to list
                else:
                    lead_serializable[segment_type] = segment_values
            segments_serializable.append(lead_serializable)

        # Saving segments as JSON string
        segments_json = json.dumps(segments_serializable)
        group.create_dataset('segments', data=np.string_(segments_json))  # Save as HDF5 string

        group.attrs['seq_len'] = seq_len
        group.attrs['fs'] = fs
        group.attrs['Source'] = source
    '''
    @staticmethod
    def load_segments(group):
        """Load segments from the HDF5 file and return them as a dict."""
        segments_json = group['segments'][()].decode('utf-8')
        return json.loads(segments_json)
    '''

# Main processor class that coordinates the ECG segmentation processing
class ECGSegmentationProcessor:
    def __init__(self, hdf5_file):
        self.loader = ECGDataLoader(hdf5_file)
        self.detector = RPeakDetector()
        self.segmenter = ECGWaveSegmenter()

    def preprocess_group(self, group_key):
        """Preprocess ECG signals for a specific group_key."""
        signals, seq_len, fs, source = self.loader.load_group_data(group_key)

        # Step 1: Detect R-peaks for all 12 leads
        all_r_peaks = self.detector.detect_r_peaks(signals, fs)

        # Step 2: Segment P-wave, QRS complex, and T-wave for all 12 leads
        all_segments = self.segmenter.segment_waves(signals, seq_len, all_r_peaks, fs)
        
        return signals, all_segments, seq_len, fs, source

    def preprocess_and_save(self, output_hdf5):
        """Load, filter, and preprocess ECG signals group by group."""
        saver = HDF5Saver(output_hdf5)
        saver.open_file()

        with h5py.File(self.loader.hdf5_file, 'r') as hdf_input:
            
            group_keys = list(hdf_input.keys())
            
            for group_key in tqdm(group_keys, desc="ECG Segmentation", unit="group"):
                signals, all_segments, seq_len, fs, source = self.preprocess_group(group_key)
                saver.save_results(group_key, signals, all_segments, seq_len, fs, source)
                del signals, all_segments

        saver.close_file()
'''
# Usage Example
if __name__ == "__main__":
    input_hdf5 = "D:/data/ECGBERT/for_git4/preprocessing/processed_ecg_data.hdf5"
    output_hdf5 = "D:/data/ECGBERT/for_git4/preprocessing/segments_ecg_data.hdf5"

    processor = ECGSegmentationProcessor(input_hdf5)
    processor.preprocess_and_save(output_hdf5)
    logger.info("ECG signal Segmentation Done.")
'''
def ECGSegmentation(base_dir):
    
    input_hdf5 = os.path.join("processed_ecg_data.hdf5")
    output_hdf5 = os.path.join("segments_ecg_data.hdf5")
    
    processor = ECGSegmentationProcessor(input_hdf5)
    processor.preprocess_and_save(output_hdf5)
    logger.info(f"Segmentation ECG data and saved to {output_hdf5}")