import numpy as np
import logging
import neurokit2 as nk
from joblib import Parallel, delayed
import json
from tqdm import tqdm

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

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

        for lead_idx in range(signals.shape[0]):
            lead_r_peaks = np.array(all_r_peaks[lead_idx])  # R-peaks for this lead
            signal_length = seq_len

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

            total_segments = np.concatenate([
                np.column_stack((segments['P'], np.full(len(segments['P']), 'P', dtype=object))),
                np.column_stack((segments['QRS'], np.full(len(segments['QRS']), 'QRS', dtype=object))),
                np.column_stack((segments['T'], np.full(len(segments['T']), 'T', dtype=object))),
                np.column_stack((segments['BG'], np.full(len(segments['BG']), 'BG', dtype=object))),
            ])

            total_segments = total_segments[np.argsort(total_segments[:, 0].astype(int))]
            segments["Total"] = total_segments

            all_segments.append(segments)

        return all_segments

# Main processor class that coordinates the ECG segmentation processing
class ECGSegmentationProcessor:
    def __init__(self, ecg_data_dict):
        """
        ecg_data_dict: Dict 형태의 ECG 데이터 입력
        ecg_data_dict 구조:
        {
            group_key: {
                "signal": np.ndarray,  # (leads, seq_len)
                "seq_len": int,
                "fs": int,
                "source": str
            }
        }
        """
        self.ecg_data_dict = ecg_data_dict
        self.detector = RPeakDetector()
        self.segmenter = ECGWaveSegmenter()

    def preprocess_group(self, group_key):
        """Preprocess ECG signals for a specific group_key."""
        group_data = self.ecg_data_dict[group_key]
        signals = group_data["signal"]
        seq_len = group_data["seq_len"]
        fs = group_data["fs"]

        # Step 1: Detect R-peaks for all 12 leads
        all_r_peaks = self.detector.detect_r_peaks(signals, fs)

        # Step 2: Segment P-wave, QRS complex, and T-wave for all 12 leads
        all_segments = self.segmenter.segment_waves(signals, seq_len, all_r_peaks, fs)
        
        return all_segments

    def preprocess(self):
        """Load, filter, and preprocess ECG signals group by group."""
        
        # 각 그룹에 대해 ECG 데이터 전처리 수행
        for group_key in tqdm(self.ecg_data_dict.keys(), desc="ECG Segmentation", unit="group"):
            all_segments = self.preprocess_group(group_key)

            # 기존의 self.ecg_data_dict에 segments 추가
            self.ecg_data_dict[group_key]["segments"] = all_segments

        #logger.info(f"ECG seg ex: {self.ecg_data_dict['00001_hr.dat']}")

        # self.ecg_data_dict 자체를 반환
        return self.ecg_data_dict

'''
def ECGSegmentation(ecg_data_dict):
    
    # ECGSegmentationProcessor 클래스 초기화 및 데이터 처리
    processor = ECGSegmentationProcessor(ecg_data_dict)

    # Preprocess and get data as a dictionary
    ecg_data_dict = processor.preprocess()
    
    for group, data in ecg_data_dict.items():
        print(f"Group: {group}, Signal Shape: {data['signals'].shape}, Segments Count: {len(data['segments'])}")
    
    return ecg_data_dict
'''