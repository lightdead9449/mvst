# src/datasets/meme_coin_data.py
import glob
import os
import re
import sqlite3
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Dict, Tuple
from .data import BaseData
import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MemeCoinRegressionData(BaseData):
    """
    Fixed implementation with proper initialization order and robust normalization.
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
        self.set_num_processes(n_proc=n_proc)
        self.config = config or {}
        
        # Initialize attributes to avoid attribute errors
        self.feature_df = None
        self.labels_df = None
        self.feature_names = []
        self.all_IDs = []
        
        # Get configuration parameters
        self.max_seq_len = int(self.config.get("max_seq_len", 128))
        self.prediction_window = int(self.config.get("prediction_window", 3))
        self.target_col = self.config.get("target", "close")
        
        # Load all data
        self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        
        # Process data and create sequences
        self._process_data()
        
        # Set up indices and features
        self._setup_indices_and_features(limit_size)
        
        # Verify data integrity
        self._verify_data()

    def _process_data(self):
        """Process and clean the raw data with proper dtype conversion"""
        # Convert timestamp
        self.all_df["timestamp"] = pd.to_datetime(self.all_df["timestamp"], unit="ms", utc=True)
        self.all_df = self.all_df.sort_values(["token", "timestamp"])
        
        # Convert all numeric columns to float32
        numeric_cols = self.all_df.select_dtypes(include=[np.number]).columns
        self.all_df[numeric_cols] = self.all_df[numeric_cols].astype(np.float32)
        
        # Calculate future returns
        self.all_df["close_fwd"] = (
            self.all_df.groupby("token")[self.target_col].shift(-self.prediction_window)
        )
        self.all_df["y"] = (self.all_df["close_fwd"] - self.all_df[self.target_col]) / self.all_df[self.target_col]
        self.all_df = self.all_df.dropna(subset=["y"])
        
        # Create sequence IDs
        self._create_sequence_ids()

    def _create_sequence_ids(self):
        """Create sequence IDs based on token and time chunks"""
        # Create a group ID for each sequence
        self.all_df["seq_group_id"] = self.all_df.groupby("token").cumcount() // self.max_seq_len
        self.all_df["sequence_id"] = self.all_df["token"] + "_" + self.all_df["seq_group_id"].astype(str)
        self.all_df = self.all_df.drop(columns=["seq_group_id"])
        
        # Filter out incomplete sequences
        sequence_counts = self.all_df["sequence_id"].value_counts()
        complete_sequences = sequence_counts[sequence_counts == self.max_seq_len].index
        self.all_df = self.all_df[self.all_df["sequence_id"].isin(complete_sequences)]

    def _setup_indices_and_features(self, limit_size):
        """Set up indices and features with proper normalization"""
        # Set index to sequence_id
        self.all_df = self.all_df.set_index("sequence_id")
        self.all_IDs = self.all_df.index.unique()
        
        # Apply size limit if specified
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]
        
        # Define feature names (exclude non-feature columns)
        exclude_cols = {
            "token", "timestamp", "close_fwd", "y", 
            "buys_volume", "sells_volume", "buys_number", "sells_number"
        }
        self.feature_names = [col for col in self.all_df.columns if col not in exclude_cols]
        
        # Initialize feature_df before normalization
        self.feature_df = self.all_df[self.feature_names].copy()
        
        # Apply normalization
        self.scaler = StandardScaler()
        self.feature_df[self.feature_names] = self.scaler.fit_transform(self.feature_df[self.feature_names])
        
        # Create labels
        self.labels_df = self.all_df.groupby("sequence_id")["y"].last().to_frame()
        
        # Verify shapes
        assert all(len(self.feature_df.loc[id]) == self.max_seq_len for id in self.all_IDs)
        assert len(self.labels_df) == len(self.all_IDs)
        
        # Normalize targets if needed (uncomment if necessary)
        # self.target_scaler = StandardScaler()
        # self.labels_df["y"] = self.target_scaler.fit_transform(self.labels_df[["y"]])

    def _verify_data(self):
        """Verify data integrity"""
        if len(self.all_IDs) == 0:
            raise ValueError("No valid sequences found - check your data and max_seq_len")
        
        if len(self.feature_names) == 0:
            raise ValueError("No features selected - check your exclude_cols")
        
        if self.feature_df.isnull().values.any():
            raise ValueError("NaN values found in features after processing")
        
        if self.labels_df.isnull().values.any():
            raise ValueError("NaN values found in labels after processing")

    def load_all(self, root_dir, file_list=None, pattern=None):
        """Load all .db files with error handling"""
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, "*.db"))
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        
        if len(data_paths) == 0:
            raise ValueError(f'No .db files found in {root_dir}')

        if pattern is not None:
            data_paths = [p for p in data_paths if re.search(pattern, os.path.basename(p))]
            if len(data_paths) == 0:
                raise ValueError(f"No files match pattern: '{pattern}'")

        if self.n_proc > 1:
            return self._load_parallel(data_paths)
        return self._load_sequential(data_paths)

    def _load_parallel(self, paths):
        """Load files in parallel using multiprocessing"""
        n_proc = min(self.n_proc, len(paths))
        logger.info(f"Loading {len(paths)} files using {n_proc} processes...")
        with Pool(processes=n_proc) as pool:
            return pd.concat(pool.map(self.load_single, paths), ignore_index=True)

    def _load_sequential(self, paths):
        """Load files sequentially"""
        return pd.concat((self.load_single(path) for path in paths), ignore_index=True)

    @staticmethod
    def load_single(filepath):
        """Load a single .db file with error handling"""
        token = os.path.basename(filepath).replace(".db", "")
        try:
            with sqlite3.connect(filepath) as conn:
                df = pd.read_sql("SELECT * FROM candles", conn)
            df["token"] = token
            return df
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {str(e)}")
            return pd.DataFrame()

    def __repr__(self):
        return (f"MemeCoinRegressionData(n_sequences={len(self.all_IDs)}, "
                f"n_features={len(self.feature_names)}, seq_len={self.max_seq_len})")
    def __getitem__(self, idx):
        sample_id = self.all_IDs[idx]
        features = self.feature_df.loc[sample_id].values.astype(np.float32)
        label = np.array([self.labels_df.loc[sample_id].values[0]] * self.max_seq_len)
        return features, label