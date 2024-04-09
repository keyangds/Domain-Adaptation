import os

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask, sample_mask_block

class Pems08(PandasDataset):
    def __init__(self):
        df, mask, dist, df_raw= self.load()
        self.df_raw = df_raw
        self.dist = dist
        super().__init__(dataframe=df, u=None, mask=mask, name='pems', freq='5T', aggr='nearest')

    def load(self, impute_zeros=True):
        data = np.load('./datasets/PEMS/pems08.npz')
        data = data['data']
        data = data.reshape((data.shape[0], data.shape[1]))
        
        df = pd.DataFrame(data)
        date_range = pd.date_range(start='2017-05-01', periods=df.shape[0], freq='5T')
        df.index = date_range
        df_raw = df
        mask = ~np.isnan(df.values)
        df.fillna(method='ffill', axis=0, inplace=True)
        dist = self.load_distance_matrix(list(df.columns))
        return df, mask, dist, df_raw

    def load_distance_matrix(self, ids):
        path = './datasets/PEMS/'
        try:
            dist = np.load(path)
        except:
            distances = pd.read_csv('./datasets/PEMS/distance_8.csv')
            num_sensors = len(ids)
            dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
            # Builds sensor id to index map.
            sensor_id_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

            # Fills cells in the matrix with distances.
            for row in distances.values:
                if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                    continue
                dist[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
            np.save(path, dist)
        return dist

    def get_similarity(self, type='dcrnn', thr=0.1, force_symmetric=False, sparse=False):
        """
        Return similarity matrix among nodes. Implemented to match DCRNN.

        :param type: type of similarity matrix.
        :param thr: threshold to increase saprseness.
        :param trainlen: number of steps that can be used for computing the similarity.
        :param force_symmetric: force the result to be simmetric.
        :return: and NxN array representig similarity among nodes.
        """
        if type == 'dcrnn':
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            adj = np.exp(-np.square(self.dist / sigma))
        elif type == 'stcn':
            sigma = 10
            adj = np.exp(-np.square(self.dist) / sigma)
        else:
            raise NotImplementedError
        adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)

        return adj

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.
        return self._mask


class MissingValuesPems08(Pems08):
    SEED = 56789
    def __init__(self, p_fault=0.0015, p_noise=0.05, fixed_mask=False):
        super(MissingValuesPems08, self).__init__()
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
            # np.save('./datasets/PEMS/pems08_mask.npy', self.eval_mask)
        else:
            self.eval_mask = np.load('./datasets/PEMS/pems08_mask.npy')

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

