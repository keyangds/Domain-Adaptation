import copy
import datetime
import os
import pathlib
import sys
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import pandas as pd 
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import fillers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool
from pytorch_lightning.utilities import CombinedLoader

def has_graph_support(model_cls):
    return model_cls in [models.GRINet, models.MPGRUNet, models.BiMPGRUNet, models.GRINODENet]

def get_model_classes(model_str):
    if model_str == 'brits':
        model, filler = models.BRITSNet, fillers.BRITSFiller
    elif model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name, fixed_mask):
    # print(dataset_name)
    # exit()

    if dataset_name[:3] == 'air':
        dataset = datasets.MissingAirSource(p_fault=0., p_noise=0.15, fixed_mask=fixed_mask)
        target_dataset = datasets.MissingAirTarget(p_fault=0., p_noise=0.15, fixed_mask=fixed_mask)
    elif dataset_name == 'discharge_point':
        # dataset = datasets.MissingValuesDischarge(p_fault=0.0015, p_noise=0.25)
        # target_dataset = datasets.MissingValuesTarget(p_fault=0.0015, p_noise=0.25)
        dataset = datasets.MissingValuesDischarge(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
        target_dataset = datasets.MissingValuesTarget(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
    elif dataset_name == 'pems':
        dataset = datasets.MissingValuesPems08(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
        target_dataset = datasets.MissingValuesPems04(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset,target_dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='grin')
    parser.add_argument("--dataset-name", type=str, default='air')
    parser.add_argument("--config", type=str, default=None)
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)
    # DA-method
    parser.add_argument('--da_method', type=str, default='None')  ### 'cotmix', 'coral', 'mmd'
    parser.add_argument('--temporal_shift', type=int, default=32)
    parser.add_argument('--mix_ratio', type=float, default=0.85)
    parser.add_argument('--aux_weight', type=float, default=1)
    parser.add_argument('--fixed_mask', type=bool, default=False)

    #### graph_constructor ### 
    parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--subgraph_size',type=int,default=1,help='k')
    parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
    parser.add_argument('--device',type=str,default='cuda:0',help='')

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args

def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    fixed_mask = args.fixed_mask
    model_cls, filler_cls = get_model_classes(args.model_name)
    dataset,target_dataset = get_dataset(args.dataset_name, fixed_mask=fixed_mask)
    da_method = args.da_method
    
    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)

    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
 
    torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                mask=dataset.training_mask,
                                eval_mask=dataset.eval_mask,
                                window=args.window,
                                stride=args.stride)
    
    torch_target_dataset = dataset_cls(*target_dataset.numpy(return_idx=True),
                                mask=target_dataset.training_mask,
                                eval_mask=target_dataset.eval_mask,
                                window=args.window,
                                stride=args.stride)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_target_dataset, **split_conf)

    #### target dataset split ### 
    #split_conf = parser_utils.filter_function_args(args, target_dataset.splitter, return_dict=True)
    #train_idxs, val_idxs, test_idxs = target_dataset.splitter(torch_target_dataset, **split_conf)

    ## no change on  the target dataset yet ### 

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
 
    dm = SpatioTemporalDataModule(torch_dataset,torch_target_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                  **data_conf)
    dm.setup()

    # if out of sample in air, add values removed for evaluation in train set
    if not args.in_sample and args.dataset_name[:3] == 'air':
        dm.torch_dataset.mask[dm.train_slice] |= dm.torch_dataset.eval_mask[dm.train_slice]

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)

    #iterables = {"source": dm, "target": dm}
    #combined_loader = CombinedLoader(iterables, mode="min_size")

    ########################################
    # predictor                            #
    ########################################

    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes,device = args.device,
                        num_nodes = 20,
                        subgraph_size = args.subgraph_size,
                        node_dim=args.node_dim,
                        buildA_true = args.buildA_true)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False),
               }
  
    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     },
                                     alpha=args.alpha,
                                     hint_rate=args.hint_rate,
                                     g_train_freq=args.g_train_freq,
                                     d_train_freq=args.d_train_freq)
    
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
  
    filler = filler_cls(**filler_kwargs)

    ########################################
    # training                             #
    ########################################
    
    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae/dataloader_idx_1', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae/dataloader_idx_1', mode='min')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         accelerator='auto',
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback])

    
    trainer.fit(filler, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    filler.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                      lambda storage, loc: storage)['state_dict'])
    filler.freeze()
    trainer.test(datamodule=dm)
    filler.eval()

    if torch.cuda.is_available():
       filler =  filler.to(args.device)

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader()['target'], return_mask=True)
    y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])  # reshape to (eventually) squeeze node channels
    # Test imputations in whole series
    eval_mask = target_dataset.eval_mask[dm.test_slice]
    df_true = target_dataset.df.iloc[dm.test_slice]
    df_true_raw = target_dataset.df_raw.iloc[dm.test_slice]

    #df_true = df_true.transform(lambda x: np.exp(x) - 1e-5)

    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mse': numpy_metrics.masked_mse,
        'mre': numpy_metrics.masked_mre,
        'mape': numpy_metrics.masked_mape,
        'nse': numpy_metrics.nse,
        'kge': numpy_metrics.kge
    }
    # Aggregate predictions in dataframes
    index = dm.torch_target_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(y_hat, index, target_dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))
   
    for aggr_by, df_hat in df_hats.items():
        # Compute error
        print(f'- AGGREGATE BY {aggr_by.upper()}')
        for metric_name, metric_fn in metrics.items():
       
            error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
            print(f' {metric_name}: {error:.4f}')
         
            #### de_transform of df_values ## 
            error_real = metric_fn(df_hat.values, df_true_raw.values, eval_mask).item()
            print(f' {metric_name}: {error_real:.4f}')

    mask = pd.DataFrame(eval_mask)
    #data_pre = (df_hat-1) * 3759.35 + 1237.58
    data_pre = df_hat
    # data_pre.to_csv('%s_pre.csv'%(args.model_name),header=None)
    # df_true_raw.to_csv('true.csv',header=None)
    # mask.to_csv('%s_mask.csv'%(args.model_name),header=None,index = None)
    # print(y_true.shape)
    return df_hats, df_true, eval_mask


if __name__ == '__main__':
    args = parse_args()
    da_method = args.da_method
    df_hats, df_true, eval_mask = run_experiment(args)

    # print(df_hats['mean'].iloc[:,5])


    # data = {
    #     'SiteID': ['4186500', '4185440', '4188496', '4182000', '4183000', '4191444', '4192574', '4184500', '4185000', '4189000', '4190000', '4192599', '4178000', '4185318', '4188100', '4191058', '4183500', '4191500', '4192500', '4193500'],
    #     'StreamLeve': [3, 4, 4, 1, 1, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 3, 1, 2, 1, 1]
    # }

    # stream_level_values = set(data['StreamLeve'])
    # index_lists = {value: [] for value in stream_level_values}

    # for index, value in enumerate(data['StreamLeve']):
    #     index_lists[value].append(index)

    # df = pd.DataFrame(data)
    # # Convert DataFrame to dictionary
    # result_dict = df.set_index('SiteID')['StreamLeve'].to_dict()

    # sub_df1 = []
    # sub_df2 = []
    # sub_df3 = []
    # sub_df4 = []

    # for SiteID in df_hats['mean'].columns.tolist():
    #     if result_dict[SiteID] == 1:
    #         sub_df1.append(SiteID)
    #     if result_dict[SiteID] == 2:
    #         sub_df2.append(SiteID)
    #     if result_dict[SiteID] == 3:
    #         sub_df3.append(SiteID)
    #     if result_dict[SiteID] == 4:
    #         sub_df4.append(SiteID)

    # hat_level1 = df_hats['mean'][sub_df1]
    # hat_level2 = df_hats['mean'][sub_df2]
    # hat_level3 = df_hats['mean'][sub_df3]
    # hat_level4 = df_hats['mean'][sub_df4]

    # true_level1 = df_true[sub_df1]
    # true_level2 = df_true[sub_df2]
    # true_level3 = df_true[sub_df3]
    # true_level4 = df_true[sub_df4]

    # eval_mask1 = eval_mask[:,index_lists[1]]
    # eval_mask2 = eval_mask[:,index_lists[2]]
    # eval_mask3 = eval_mask[:,index_lists[3]]
    # eval_mask4 = eval_mask[:,index_lists[4]]    

    # metrics = {
    #     'mae': numpy_metrics.masked_mae,
    #     'mse': numpy_metrics.masked_mse,
    #     'mre': numpy_metrics.masked_mre,
    #     'mape': numpy_metrics.masked_mape,
    #     'nse': numpy_metrics.nse,
    #     'kge': numpy_metrics.kge
    # }

    # stream_result_path = 'stream_result_' + da_method + '_point.txt'
    # node_result_path = 'node_result_' + da_method + '_point.txt'

    # with open(stream_result_path, 'a') as file:

    #     file.write("####### STREAM LEVEL 1 ########\n")
    #     for metric_name, metric_fn in metrics.items():
    #         error = metric_fn(hat_level1.values, true_level1.values, eval_mask1).item()
    #         file.write(f' {metric_name}: {error:.4f}')
    #     file.write("\n")

    #     file.write("####### STREAM LEVEL 2 ########\n")
    #     for metric_name, metric_fn in metrics.items():
    #         error = metric_fn(hat_level2.values, true_level2.values, eval_mask2).item()
    #         file.write(f' {metric_name}: {error:.4f}')
    #     file.write("\n")

    #     file.write("####### STREAM LEVEL 3 ########\n")
    #     for metric_name, metric_fn in metrics.items():
    #         error = metric_fn(hat_level3.values, true_level3.values, eval_mask3).item()
    #         file.write(f' {metric_name}: {error:.4f}')
    #     file.write("\n")

    #     file.write("####### STREAM LEVEL 4 ########\n")
    #     for metric_name, metric_fn in metrics.items():
    #         error = metric_fn(hat_level4.values, true_level4.values, eval_mask4).item()
    #         file.write(f' {metric_name}: {error:.4f}')
        
    #     file.write("\n")

    # with open(node_result_path, 'a') as file:
    #     for i in range(20):
    #         file.write("####### Node " + str(i) + " ########\n")
    #         for metric_name, metric_fn in metrics.items():
    #             error = metric_fn(df_hats['mean'].iloc[:,i].values, df_true.iloc[:,i].values, eval_mask[:,i]).item()
    #             file.write(f' {metric_name}: {error:.4f}')
        
    #         file.write("\n")

    # np.save(da_method + '_hat_point.npy', df_hats['mean'].values)
    # np.save(da_method + 'cotmix_true_point.npy', df_true.values)
    # np.save(da_method + 'cotmix_mask_point.npy', eval_mask)
            


    