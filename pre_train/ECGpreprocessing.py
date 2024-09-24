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
            # Mask NaN and Inf values
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

        # Check for NaN or Inf and replace with original signal if found
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

        # Check for NaN or Inf and replace with original signal if found
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
        # 결측치 처리 (신호 필터링 전에 처리)
        signals = np.array([self.missing_value_handler.fill_missing(sig) for sig in signals])

        filtered_signals = signals
        for bandstop_filter in self.filters:
            filtered_signals = np.array([bandstop_filter.apply(sig) for sig in filtered_signals])
        
        processed_signals = np.array([self.baseline_removal.apply(sig) for sig in filtered_signals])

        # 마지막 후처리: 여전히 NaN 또는 Inf 값이 있다면 원래 신호로 대체
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
        """Preprocess a single group of ECG data."""
        fs = group_data.attrs['fs']
        ecg_signals = group_data['signal'][:]  # Load signal data from group

        # Initialize the signal processor
        signal_processor = ECGSignalProcessor(fs)

        # Apply filters and remove baseline wander
        processed_signals = signal_processor.process(ecg_signals)
        return processed_signals, fs, group_data.attrs['seq_len'], group_data.attrs['Source']
    
    def preprocess_and_save(self, output_hdf5: str):
        """Load, filter, and preprocess ECG signals one group at a time."""
        with h5py.File(self.hdf5_file, 'r') as hdf_input, h5py.File(output_hdf5, 'w') as hdf_output:
            for group_key in hdf_input.keys():
                #logger.info(f"Processing group {group_key}")
                group_data = hdf_input[group_key]
                
                # Preprocess the data for this group
                processed_signals, fs, seq_len, source = self.preprocess_group(group_data)
                
                # Save the processed signals to the output HDF5 file
                group = hdf_output.create_group(group_key)
                group.create_dataset('signal', data=processed_signals)
                group.attrs['fs'] = fs
                group.attrs['seq_len'] = seq_len
                group.attrs['Source'] = source

'''
# Usage Example
if __name__ == "__main__":
    input_hdf5 = "D:/data/ECGBERT/for_git4/preprocessing/ecg_data2.hdf5"
    output_hdf5 = "D:/data/ECGBERT/for_git4/preprocessing/processed_ecg_data.hdf5"
    
    preprocessor = ECGPreprocessor(input_hdf5)
    preprocessor.preprocess_and_save(output_hdf5)
'''
def ECGPreprocessing(base_dir):
    
    input_hdf5 = os.path.join(base_dir, "ecg_data.hdf5")
    output_hdf5 = os.path.join(base_dir, "processed_ecg_data.hdf5")
    
    preprocessor = ECGPreprocessor(input_hdf5)
    preprocessor.preprocess_and_save(output_hdf5)
    
    logger.info(f"Processed ECG data saved to {output_hdf5}")
