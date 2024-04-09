import os
from argparse import ArgumentParser
import sys
import numpy as np
import pytorch_lightning as pl
import torchmetrics
import torch
import yaml
import torch.nn.functional as F
import pandas as pd 
import argparse
from dataframe import DataFrameDataset, CombinedDataLoader
from torch import nn
from lib import datasets, fillers
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_model(model_str):
    if model_str == 'rnn':
        model, filler = models.RNNImputer, 'filler'
    elif model_str == 'tcn':
        model, filler = models.TCNImputer, 'filler'
    elif model_str == 'brits':
        model, filler = models.BRITSNet, 'britsfiller'
    elif model_str == 'transformer':
        model, filler = models.TransformerImputer, 'filler'
    else:
        raise ValueError(f'Model {model_str} not available.')
 
    return model, filler

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = ArgumentParser()
    # Architecture and Dataset
    parser.add_argument("--model_name", type=str, default='rnn')
    parser.add_argument("--dataset_name", type=str, default='air')
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fixed_mask", type=str2bool, default=False)
    parser.add_argument('--scale', type=str2bool, default=True)
    parser.add_argument('--scaling_axis', type=str, default='channels')
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    # Training Params
    parser.add_argument('--training', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--use-lr-schedule', type=str2bool, default=True)
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--device',type=str,default='cuda:0',help='')
    # DA-Method
    parser.add_argument("--da_method", type=str, default=None) ### 'direct', 'coral', 'cotmix', 'dann', 'dirt', 'advSKM'
    parser.add_argument('--temporal_shift', type=int, default=32)
    parser.add_argument('--mix_ratio', type=float, default=0.7)
    parser.add_argument('--aux_weight', type=float, default=1)

    known_args, _ = parser.parse_known_args()
 

    # args = parser.parse_args([])
    # for arg, value in vars(args).items():
    #     print(f"{arg}: {value}")
    # exit()
    args = parser.parse_args()


    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    model_cls, filler_name = get_model(args.model_name)
    parser = model_cls.add_model_specific_args(parser)

    return args, filler_name

def get_dataset(dataset_name, fixed_mask=False):
    if dataset_name[:3] == 'air':
        dataset = datasets.MissingAirSource(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
        target_dataset = datasets.MissingAirTarget(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
    elif dataset_name == 'discharge':
        dataset = datasets.MissingValuesDischarge(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
        target_dataset = datasets.MissingValuesTarget(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
    elif dataset_name == 'pems':
        dataset = datasets.MissingValuesPems08(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
        target_dataset = datasets.MissingValuesPems04(p_fault=0., p_noise=0.25, fixed_mask=fixed_mask)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset,target_dataset

if __name__ == '__main__':
    args, filler_name = parse_args()

    source_dataset, target_dataset = get_dataset(args.dataset_name, fixed_mask=args.fixed_mask)

    source_split = source_dataset.splitter()
    target_split = target_dataset.splitter()
   
    args_dict = vars(args)
    valid_params = ['dataset', 'split', 'scale', 'scaling_axis', 'batch_size']
    # Filter args_dict to keep only the keys that are valid for DataFrameDataset
    filtered_args = {k: v for k, v in args_dict.items() if k in valid_params}

    source = DataFrameDataset(source_dataset, source_split, **filtered_args)
    target = DataFrameDataset(target_dataset, target_split, **filtered_args)
    
    source_train_loader, source_val_loader, source_test_loader = source.setup()
    target_train_loader, target_val_loader, target_test_loader = target.setup()

    train_dataloader = CombinedDataLoader(source_train_loader, target_train_loader)
    val_dataloader = CombinedDataLoader(source_val_loader, target_val_loader)

    # LOSS and METRICS
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False),
               }

    ##### Model
    ##### MODEL PARAMETER and FILLER PARAMETER
    if args.model_name == 'rnn':
        model_cls = models.RNNImputer
        model_kwargs = {'d_in': source.d_in, 'd_model': args.d_model}
    elif args.model_name == 'tcn':
        model_cls = models.TCNImputer
        model_kwargs = {'input_size': args.input_size, 'output_size': args.output_size,
                        'num_channels': args.num_channels, 'kernel_size': args.kernel_size}
    elif args.model_name == 'transformer':
        model_cls = models.TransformerImputer
        model_kwargs = {'d_in': args.input_size, 'd_model': args.d_model,
                        'nhead': args.nhead, 'num_encoder_layers': args.num_encoder_layers,
                        'dim_feedforward': args.dim_feedforward, 'dropout': args.dropout}
    elif args.model_name == 'brits':
        model_cls = models.BRITSNet
        model_kwargs = {'d_in': source.d_in, 'd_hidden': args.d_hidden}


    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    filler_params = {
        'batch_size': args.batch_size,
        'input_size': args.input_size,
        'd_in': source.d_in,
        'd_hidden': args.d_hidden,
        'optim_class': torch.optim.Adam,
        'optim_kwargs': {'lr': args.lr, 'weight_decay': args.l2_reg},
        'loss_fn': loss_fn,
        'metrics': metrics,
        'scheduler_class': scheduler_class,
        'scheduler_kwargs': {'eta_min': 0.0001, 'T_max': args.epoch},
        'da_method': args.da_method,
        'mix_ratio': args.mix_ratio,
        'temporal_shift': args.temporal_shift,
        'device': args.device,
        'loader_size': len(train_dataloader)
    }

    if filler_name == 'filler':
        filler = fillers.Filler(model_cls, model_kwargs, **filler_params)
    elif filler_name == 'britsfiller':
    
        filler = fillers.BRITSFiller(model_cls, model_kwargs, **filler_params)
       
    # Ensure the directory exists
    save_dir = os.path.join('./trained_model', args.dataset_name, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    if args.training:
        trainer = pl.Trainer(max_epochs=args.epoch,
                            accelerator='auto',
                            gradient_clip_val=args.grad_clip_val,
                            gradient_clip_algorithm=args.grad_clip_algorithm,
                            val_check_interval=1)

        trainer.fit(filler, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Save the trained model
        trainer.save_checkpoint(os.path.join(save_dir, f"{args.da_method}_trained_model.ckpt"))

    if torch.cuda.is_available():
        device = torch.device(args.device)
        filler.model.to(device)
    else:
        device = torch.device('cpu')

    checkpoint_path = os.path.join(save_dir, f"{args.da_method}_trained_model.ckpt")
   
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        new_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        filler.model.load_state_dict(new_state_dict, strict=False)

    filler.freeze()
    filler.eval()
    
    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(target_test_loader, return_mask=True)
    
    y_true_rescaled = target.scaler.inverse_transform(y_true)
    y_hat_rescaled = target.scaler.inverse_transform(y_hat)

    final_y_true = np.zeros(target_dataset.df.iloc[target_split[2]].shape)
    final_y_hat = np.zeros(target_dataset.df.iloc[target_split[2]].shape)

    sequence_length = 24
    num_sequences = len(y_true_rescaled)

    # Assuming the predictions align with the windows starting from the first index of the test data
    for i in range(num_sequences):
        start_idx = i
        end_idx = start_idx + sequence_length
        
        # Ensure not to exceed the bounds of the final arrays
        end_idx = min(end_idx, final_y_true.shape[0])
        
        # Assuming y_true_rescaled and y_hat_rescaled are numpy arrays with shape (num_sequences, sequence_length, num_features)
        # And that you want to update only the part of the sequence that corresponds to the original data indices
        final_y_true[start_idx:end_idx] = y_true_rescaled[i][:end_idx-start_idx]
        final_y_hat[start_idx:end_idx] = y_hat_rescaled[i][:end_idx-start_idx]
        final_mask = target_dataset.eval_mask[target_split[2]]

    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mse': numpy_metrics.masked_mse,
        'mre': numpy_metrics.masked_mre,
        'mape': numpy_metrics.masked_mape
        # 'nse': numpy_metrics.nse,
        # 'kge': numpy_metrics.kge
    }

    for metric_name, metric_fn in metrics.items():
        error = metric_fn(final_y_hat, final_y_true, final_mask).item()
        print(f' {metric_name}: {error:.4f}')
        
        #### de_transform of df_values ## 
        error_real = metric_fn(final_y_hat, target_dataset.df.iloc[target_split[2]].values, final_mask).item()
        print(f' {metric_name}: {error_real:.4f}')
    









