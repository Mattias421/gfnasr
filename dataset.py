import torch
from torch.utils.data import Dataset
import h5py
import os
from pathlib import Path

class HDF5Dataset(Dataset):
    """
    A PyTorch Dataset to load Whisper encoder embeddings from an HDF5 file.
    Assumes the HDF5 file contains:
    - A dataset (e.g., 'log_mel_features' or 'encoder_embeddings') with shape (num_samples, seq_len, embed_dim).
    - Optionally, a dataset named 'filenames' or 'relative_paths' for reference.
    """
    def __init__(self,
                 h5_file_path: str,
                 dataset_name: str = "encoder_embeddings", # Name of the main embedding dataset in HDF5
                 transform=None,
                 target_device: str = 'cpu',
                 load_paths_dataset_name: str = "relative_filepaths" # Optional: name of dataset holding file paths
                ):
        """
        Args:
            h5_file_path (str): Path to the HDF5 file.
            dataset_name (str): Name of the dataset within the HDF5 file containing the embeddings.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_device (str or torch.device): The device to move the loaded tensor to.
            load_paths_dataset_name (str, optional): Name of the dataset holding original file paths, if any.
        """
        super().__init__()
        self.h5_file_path = h5_file_path
        self.dataset_name = dataset_name
        self.transform = transform
        self.target_device = torch.device(target_device)
        self.load_paths_dataset_name = load_paths_dataset_name

        if not os.path.exists(self.h5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file_path}")

        # Open the HDF5 file once to get shape and dtype, but dataset access will be lazy.
        # HDF5 handles lazy loading of slices efficiently.
        with h5py.File(self.h5_file_path, 'r') as f:
            if self.dataset_name not in f:
                raise ValueError(f"Dataset '{self.dataset_name}' not found in HDF5 file '{self.h5_file_path}'. "
                                 f"Available datasets: {list(f.keys())}")
            self.dataset_shape = f[self.dataset_name].shape
            self.dataset_dtype = f[self.dataset_name].dtype # NumPy dtype
            self.num_samples = self.dataset_shape[0]

            if self.load_paths_dataset_name:
                if self.load_paths_dataset_name not in f:
                    print(f"Warning: Paths dataset '{self.load_paths_dataset_name}' not found in HDF5 file.")
                    self.paths_available = False
                else:
                    self.paths_available = True
            else:
                self.paths_available = False

        # The h5py.File object should NOT be kept open as a class member if using
        # multiprocessing (num_workers > 0 in DataLoader). Each worker needs its own file handle.
        # Instead, we will open it in __getitem__.

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Loads the data for the given index from the HDF5 file.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_samples} samples.")

        # Open HDF5 file here for each item. h5py is thread-safe but file objects are not.
        # For multiprocessing, each worker process needs its own file handle.
        # This might seem inefficient, but h5py is optimized for this.
        # Re-opening is generally safe and recommended for multi-worker DataLoaders.
        with h5py.File(self.h5_file_path, 'r') as f:
            # Accessing the slice reads it from disk into RAM (for this slice only)
            embedding_np = f[self.dataset_name][idx] # This loads the data for the specific index

            path_info = None
            if self.paths_available and self.load_paths_dataset_name:
                try:
                    path_info_bytes = f[self.load_paths_dataset_name][idx]
                    # HDF5 string datasets are often fixed-length byte strings
                    if isinstance(path_info_bytes, bytes):
                        path_info = path_info_bytes.decode('utf-8').strip('\x00') # Decode and strip null chars
                    else: # If it's already a string (e.g., variable-length UTF-8 string dataset)
                        path_info = str(path_info_bytes)


                    wav_len = f["wav_lens"][idx]
                    wav_len = torch.tensor(wav_len, dtype=torch.long)
                except Exception as e:
                    print(f"Warning: Could not load path info for index {idx}: {e}")
                    path_info = "ErrorLoadingPath"


        # Convert NumPy array to PyTorch tensor
        # The dtype from HDF5 (self.dataset_dtype) is a NumPy dtype.
        # torch.from_numpy will infer the correct PyTorch dtype.
        embedding_tensor = torch.from_numpy(embedding_np).to(self.target_device)

        if self.transform:
            embedding_tensor = self.transform(embedding_tensor)

        if path_info is not None:
            path_info = Path(path_info).stem
            return embedding_tensor, path_info, wav_len # Return embedding and its original path
        else:
            return embedding_tensor # Just return the embedding
