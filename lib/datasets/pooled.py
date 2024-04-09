import os
import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask,sample_mask_block


class Target(PandasDataset):
    def __init__(self):
        df,  mask,df_raw = self.load()
        super().__init__(dataframe=df, u=None, mask=mask, name='target', freq='1D', aggr='nearest')
        self.df_raw = df_raw

    def load(self, impute_zeros=True):
        #path = os.path.join(datasets_path['discharge'], 'SSC_discharge.csv')
        df = pd.read_csv('./datasets/discharge/SSC_pooled.csv',index_col=0)

        df.index = pd.to_datetime(df.index)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1D')
        df = df.reindex(index=date_range) 
        df = df.loc['2015/4/15':'2021/9/9',:]
        df_raw = df
        mask = ~np.isnan(df.values)
        df.fillna(method='ffill', axis=0, inplace=True)
        return df.astype('float32'),  mask.astype('uint8'), df_raw.astype('float32')

    def get_similarity(self, thr=0.1, force_symmetric=False, sparse=False):
        adj = np.array(pd.read_csv('./datasets/discharge/SSC_sites_flow_direction.csv',index_col=0).values)
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

class MissingValuesTarget(Target):
    SEED = 56789
    def __init__(self, p_fault=0.0015, p_noise=0.05, fixed_mask=False):
        super(MissingValuesTarget, self).__init__()
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        self.fixed_mask = fixed_mask
        self.train_index = 2110
       
        self.test_index = 230
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
            # np.save('./datasets/discharge/pooled_mask.npy', self.eval_mask)
        else:
            self.eval_mask = np.load('./datasets/discharge/pooled_mask.npy')
        # print(self.eval_mask.shape)
        # print(np.count_nonzero(self.eval_mask == 1))
        # print(np.count_nonzero(self.eval_mask == 0))
        # exit()
  
    @property
    def training_mask(self):
   
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, window=0):
        idx = np.arange(len(self.df))
    
        test_len = 180
        val_len = 50

        test_start = len(idx) - test_len
        val_start = test_start - val_len

        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]

