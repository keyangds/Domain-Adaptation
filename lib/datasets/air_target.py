import os

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils.utils import disjoint_months, infer_mask, compute_mean, geographical_distance, thresholded_gaussian_kernel
from ..utils import sample_mask, sample_mask_block

class AirTarget(PandasDataset):
    SEED = 3210
    def __init__(self, impute_nans=False, small=False, freq='60T', masked_sensors=None):
        self.random = np.random.default_rng(self.SEED)
        self.test_months = [3, 6, 9, 12]
        self.eval_mask = None
        df, mask, dist, df_raw= self.load()
        self.df_raw = df_raw

        self.dist = dist
        if masked_sensors is None:
            self.masked_sensors = list()
        else:
            self.masked_sensors = list(masked_sensors)
        super().__init__(dataframe=df, u=None, mask=mask, name='air', freq=freq, aggr='nearest')
        
    def load(self):
        eval_mask = None
        df = pd.read_csv('./datasets/air_quality/tianjin.csv',index_col=0)
        df.index = pd.to_datetime(df.index)
        df_raw = df
        # stations = pd.DataFrame(pd.read_hdf(path, 'stations'))
        mask = ~np.isnan(df.values)
        df.fillna(method='ffill', axis=0, inplace=True)
        stations = pd.read_csv('./datasets/air_quality/station_t.csv',index_col=0)
        # compute distances from latitude and longitude degrees
        st_coord = stations.loc[:, ['latitude', 'longitude']]
        dist = geographical_distance(st_coord, to_rad=True).values
        return df, mask, dist, df_raw

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.
        return self._mask

    def get_similarity(self, thr=0.1, include_self=False, force_symmetric=False, sparse=False, **kwargs):
        theta = np.std(self.dist[:27, :27])  # use same theta for both air and air36
        adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj



class MissingAirTarget(AirTarget):
    SEED = 56789
    def __init__(self, p_fault=0.0015, p_noise=0.05, fixed_mask=False):
        super(MissingAirTarget, self).__init__()
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        self.fixed_mask = fixed_mask
        self.train_index = int(self.df.shape[0]*0.8)
        self.test_index = self.df.shape[0] - int(self.df.shape[0]*0.8)
        self.nodes = self.df.shape[1]
        self.n_channels = self.df.shape[-1]
        
        eval_mask = sample_mask(self.mask[0:self.train_index,:], p=self.p_noise)
        eval_mask_block = sample_mask_block(self.mask[-self.test_index:,:].shape,
                                self.p_fault,
                                self.p_noise,
                                min_seq=5,
                                max_seq=15,
                                rng=self.rng)
        if self.fixed_mask == False:
            self.eval_mask = np.concatenate((eval_mask,eval_mask_block),axis=0)
            np.save('./datasets/air_quality/tianjin_mask.npy', self.eval_mask)
        else:
            self.eval_mask = np.load('./datasets/air_quality/tianjin_mask.npy')
   
    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, window=0):
        idx = np.arange(len(self.df))
    
        test_len = int((self.test_index)*0.8)
        val_len = self.test_index - test_len

        test_start = len(idx) - test_len
        val_start = test_start - val_len


        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]

