import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import StandardScaler, MinMaxScaler, LogScaler
from imputation import ImputationDataset
import pandas as pd
import numpy as np

class DataFrameDataset(Dataset):
    def __init__(self, dataset, idx, scale, scaling_axis, scaling_type='std', batch_size = 32, workers=1,
                 windows=24, horizon=24):
        # Assuming all columns are features
        self.dataset = dataset
        self.idx = idx
        self.scale = scale
        self.scaling_axis = scaling_axis
        self.scaling_type = scaling_type
        self.batch_size = batch_size
        self.workers = workers
        self.windows = windows
        self.horizon = horizon
        self.scaler = None
        ##### data
        self.trainset = dataset.df.iloc[idx[0]].values
        self.valset = dataset.df.iloc[idx[1]].values
        self.testset = dataset.df.iloc[idx[2]].values
        ##### ground truth
        self.train_features = dataset.df.iloc[idx[0]].values
        self.val_features = dataset.df.iloc[idx[1]].values
        self.test_features = dataset.df.iloc[idx[2]].values
        ##### mask
        self.train_mask = dataset.eval_mask[idx[0]]
        self.val_mask = dataset.eval_mask[idx[1]]
        self.test_mask = dataset.eval_mask[idx[2]]
        

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        # Convert features to PyTorch tensors
        data_idx = self.idx[idx]  # Fetching the actual data index
        features = torch.tensor(self.features[data_idx], dtype=torch.float)
        mask = torch.tensor(self.mask[data_idx], dtype=torch.bool)
        # Returning features and mask; adjust as necessary to include targets or other data
        return features, mask

    def get_scaling_axes(self, dim):
        scaling_axis = tuple()
        if dim == 'global':
            scaling_axis = (0, 1, 2)
        elif dim == 'channels':
            scaling_axis = (0, 1)
        elif dim == 'nodes':
            scaling_axis = (0,)
        # Remove last dimension for temporal datasets

        if not len(scaling_axis):
            raise ValueError(f'Scaling axis "{self.dim}" not valid.')

        return scaling_axis

    def get_scaler(self):
        if self.scaling_type == 'std':
            return StandardScaler
        elif self.scaling_type == 'minmax':
            return MinMaxScaler
        elif self.scaling_type == 'log':
            return LogScaler
        else:
            return NotImplementedError
    
    def setup(self):
        if self.scale:
            scaling_axis = self.get_scaling_axes(self.scaling_axis)
            self.scaler = self.get_scaler()(scaling_axis).fit(self.train_features, mask=self.train_mask, keepdims=True).to_torch()
            
            train_data = self.scaler.transform(torch.from_numpy(self.trainset))
            train_features = self.scaler.transform(torch.from_numpy(self.train_features))
            train_mask = torch.tensor(self.train_mask)

            val_data = self.scaler.transform(torch.from_numpy(self.valset))
            val_features = self.scaler.transform(torch.from_numpy(self.val_features))
            val_mask = torch.tensor(self.val_mask)

            test_data = self.scaler.transform(torch.from_numpy(self.testset))
            test_features = self.scaler.transform(torch.from_numpy(self.test_features))
            test_mask = torch.tensor(self.test_mask)

            train_loader = self._data_loader(train_data, train_features, train_mask)
            val_loader = self._data_loader(val_data, val_features, val_mask)
            test_loader = self._data_loader(test_data, test_features, test_mask)
            
        return train_loader, val_loader, test_loader

    def _data_loader(self, x, y, mask, shuffle=False, batch_size=None, **kwargs):
        batch_size = self.batch_size if batch_size is None else batch_size
        dataset = ImputationDataset(x, y, mask)
        dataloader = DataLoader(dataset,
                                shuffle=shuffle,
                                batch_size=batch_size,
                                num_workers=self.workers,
                                **kwargs)
        return dataloader
    
    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--scaling-axis', type=str, default="channels")
        parser.add_argument('--scaling-type', type=str, default="std")
        parser.add_argument('--scale', type=bool, default=True)
        parser.add_argument('--workers', type=int, default=0)
        parser.add_argument('--samples-per-epoch', type=int, default=None)
        return parser
    
    @property
    def n_nodes(self):
        return self.dataset.nodes 

    @property
    def d_in(self):
        #if not self.has_setup_fit:
            #raise ValueError('You should initialize the datamodule first.')
        return self.dataset.n_channels

    @property
    def d_out(self):
        #if not self.has_setup_fit:
            #raise ValueError('You should initialize the datamodule first.')
        return self.torch_dataset.horizon
    
class CombinedDataLoader:
    def __init__(self, source_loader, target_loader):
        self.source_loader = source_loader
        self.target_loader = target_loader

    def __iter__(self):
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)
        return self

    def __next__(self):
        try:
            source_batch = next(self.source_iter)
        except StopIteration:
            # Reset source iterator if it runs out of data
            self.source_iter = iter(self.source_loader)
            source_batch = next(self.source_iter)
        
        try:
            target_batch = next(self.target_iter)
        except StopIteration:
            # Reset target iterator if it runs out of data
            self.target_iter = iter(self.target_loader)
            target_batch = next(self.target_iter)
        
        return {'source': source_batch, 'target': target_batch}

    def __len__(self):
        return max(len(self.source_loader), len(self.target_loader))
