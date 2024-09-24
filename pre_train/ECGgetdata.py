import os
import pandas as pd
import h5py
from wfdb import rdrecord
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# Class for finding ECG files in a directory
class ECGFileFinder:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def find_files(self, extensions=('.mat', '.dat')):
        """Find files with specific extensions in the base directory."""
        ecg_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.base_dir)
            for file in files
            if file.endswith(extensions)
        ]
        logger.info(f"Found {len(ecg_files)} ECG files.")
        return ecg_files

# Class for loading ECG data from a file
class ECGDataLoader:
    @staticmethod
    def load_data(file_path):
        """Load ECG data from a .mat or .dat file."""
        try:
            record = rdrecord(file_path.replace('.mat', '').replace('.dat', ''))
            ecg_data = record.p_signal.T  # Transpose to have each signal as a row
            fs = record.fs
            seq_len = ecg_data.shape[1]  # Length of each signal sequence
            #logger.info(f"Loaded data from {file_path}")
            return ecg_data, fs, seq_len
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

# Class for saving ECG data into HDF5 format
class HDF5Saver:
    def __init__(self, output_file):
        self.output_file = output_file
        self.hdf = None

    def open_file(self):
        """Open HDF5 file for writing."""
        self.hdf = h5py.File(self.output_file, 'w')
        #logger.info(f"HDF5 file created: {self.output_file}")

    def close_file(self):
        """Close the HDF5 file."""
        if self.hdf:
            self.hdf.close()
            #logger.info(f"HDF5 file saved: {self.output_file}")

    def save_data(self, file_name, ecg_data, fs, seq_len):
        """Save the ECG data and metadata to an HDF5 group."""
        file_group = self.hdf.create_group(file_name)
        file_group.create_dataset('signal', data=ecg_data)
        file_group.attrs['fs'] = fs
        file_group.attrs['seq_len'] = seq_len
        file_group.attrs['Source'] = file_name
        #logger.info(f"Data saved for file: {file_name}")

# Main processor class that coordinates file processing and saving
class ECGDataProcessor:
    def __init__(self, base_dir, chunk_size=5000):
        self.chunk_size = chunk_size
        self.finder = ECGFileFinder(base_dir)
        self.loader = ECGDataLoader()

    def process_files(self, output_hdf5):
        """Process ECG files and save them in HDF5 format."""
        ecg_files = self.finder.find_files()
        hdf5_saver = HDF5Saver(output_hdf5)
        hdf5_saver.open_file()

        for i in range(0, len(ecg_files), self.chunk_size):
            chunk_files = ecg_files[i:i + self.chunk_size]
            for file in chunk_files:
                try:
                    ecg_data, fs, seq_len = self.loader.load_data(file)
                    hdf5_saver.save_data(os.path.basename(file), ecg_data, fs, seq_len)
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")

        hdf5_saver.close_file()
'''
# Usage
if __name__ == "__main__":
    base_dir = "D:/data/ECGBERT/org_data/pre_train"
    output_hdf5 = "D:/data/ECGBERT/for_git4/preprocessing/ecg_data2.hdf5"
    
    processor = ECGDataProcessor(base_dir)
    processor.process_files(output_hdf5)
    
    logger.info("Get Data pkl to hdf5")
'''
# 1. Get Data - pkl to hdf5
def ECGGetData(org_dir, output_dir):

    output_hdf5 = os.path.join(output_dir, 'ecg_data.hdf5')
    
    processor = ECGDataProcessor(org_dir)
    processor.process_files(output_hdf5)
    logger.info(f"Get ECG data and saved to {output_hdf5}")
