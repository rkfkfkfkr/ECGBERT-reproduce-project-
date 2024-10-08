import h5py
import numpy as np
import logging
from scipy.signal import butter, filtfilt
import pywt
import os

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# Interpolation MissingValue
class MissingValueHandler:
    def __init__(self, method='linear'):
        self.method = method

    def fill_missing(self, signal: np.ndarray):
        """Fill missing values in the signal using interpolation or other methods."""
        if np.isnan(signal).any() or np.isinf(signal).any():
            logger.warning("Missing values detected. Applying interpolation.")
            mask = np.isnan(signal) | np.isinf(signal)
            not_nan_inf = ~mask
            valid_indices = np.where(not_nan_inf)[0]
            
            if len(valid_indices) == 0:
                logger.error("No valid values in the signal to interpolate. Returning original signal.")
                return signal
            
            valid_signal = signal[not_nan_inf]
            interpolated_signal = np.interp(np.arange(len(signal)), valid_indices, valid_signal)
            
            return interpolated_signal
        
        return signal

# Bandstop Filter Class
class BandstopFilter:
    def __init__(self, fs: int, lowcut: float, highcut: float, order: int = 2):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def apply(self, signal: np.ndarray):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='bandstop', fs=self.fs)
        filt_sig = filtfilt(b, a, signal)

        if np.isnan(filt_sig).any() or np.isinf(filt_sig).any():
            logger.warning("NaN or Inf detected in BandstopFilter. Using original signal.")
            filt_sig = signal

        return filt_sig

# Wavelet Baseline Wander Removal Class
class BaselineWanderRemoval:
    def __init__(self, wavelet: str = 'db4', level: int = 2):
        self.wavelet = wavelet
        self.level = level

    def apply(self, signal: np.ndarray):
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        coeffs[0] = np.zeros_like(coeffs[0])  # Remove baseline wander
        filt_sig = pywt.waverec(coeffs, self.wavelet)

        if np.isnan(filt_sig).any() or np.isinf(filt_sig).any():
            logger.warning("NaN or Inf detected in BaselineWanderRemoval. Using original signal.")
            filt_sig = signal

        return filt_sig

# ECG Signal Processor
class ECGSignalProcessor:
    def __init__(self, fs: int, missing_value_handler=None):
        self.fs = fs
        self.filters = [
            BandstopFilter(fs, 50, 60)
        ]
        self.baseline_removal = BaselineWanderRemoval()
        self.missing_value_handler = missing_value_handler or MissingValueHandler()

    def process(self, signals: np.ndarray):
        signals = np.array([self.missing_value_handler.fill_missing(sig) for sig in signals])

        filtered_signals = signals
        for bandstop_filter in self.filters:
            filtered_signals = np.array([bandstop_filter.apply(sig) for sig in filtered_signals])
        
        processed_signals = np.array([self.baseline_removal.apply(sig) for sig in filtered_signals])

        for idx, sig in enumerate(processed_signals):
            if np.isnan(sig).any() or np.isinf(sig).any():
                logger.warning(f"Final NaN or Inf detected in signal {idx}. Reverting to original signal.")
                processed_signals[idx] = signals[idx]

        return processed_signals

# ECG Preprocessor Class
class ECGPreprocessor:
    def __init__(self, hdf5_file: str):
        self.hdf5_file = hdf5_file

    def preprocess_group(self, group_data):
        """Preprocess a single group of ECG data and return processed data."""
        fs = group_data.attrs['fs']
        ecg_signals = group_data['signal'][:]  # Load signal data from group

        # Initialize the signal processor
        signal_processor = ECGSignalProcessor(fs)

        # Apply filters and remove baseline wander
        processed_signals = signal_processor.process(ecg_signals)
        return processed_signals, fs, group_data.attrs['seq_len'], group_data.attrs['Source']
    
    def preprocess(self):
        """Preprocess ECG signals from the HDF5 file and return as a dictionary."""
        all_data = {}

        with h5py.File(self.hdf5_file, 'r') as hdf_input:
            for group_key in hdf_input.keys():
                group_data = hdf_input[group_key]
                
                # Preprocess the data for this group
                processed_signals, fs, seq_len, source = self.preprocess_group(group_data)
                
                # Save processed data to a dictionary
                all_data[group_key] = {
                    'signal': processed_signals,
                    'fs': fs,
                    'seq_len': seq_len,
                    'source': source
                }

        return all_data

""" 
all_data = {
    "group_key" : { 'signal' : np.ndarray() # (lead, seq_len)
                    'fs' : 500
                    'seq_len': int
                    'source' : str #filename }
}

"""

'''
def ECGPreprocessing(input_hdf5):
    
    preprocessor = ECGPreprocessor(input_hdf5)
    
    # 데이터를 HDF5 파일에 저장하지 않고 메모리 내에서 반환
    ecg_data_dict = preprocessor.preprocess()
    
    # 결과 확인
    for group, data in ecg_data_dict.items():
        print(f"Group: {group}")
        print(f"Signal Shape: {data['signal'].shape}")
        print(f"Sampling Rate: {data['fs']}, Sequence Length: {data['seq_len']}")
        print(f"Source: {data['Source']}\n")
        
    return ecg_data_dict
'''