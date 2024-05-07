import inspect
from copy import deepcopy
from torch import nn
import pytorch_lightning as pl
import torch
#from pytorch_lightning.core.decorators import auto_move_data
#from pytorch_lightning.metrics import MetricCollection
from torchmetrics import MetricCollection
from pytorch_lightning.utilities import move_data_to_device
from ..DA.models import ReverseLayerF, Discriminator, Discriminator_CDAN, RandomLayer, AdvSKM_Disc, tf_encoder, tf_decoder, SpectralConv1d

from .. import epsilon
from ..nn.utils.metric_base import MaskedMetric
from ..utils.utils import ensure_list
from ..DA.loss import SupConLoss, ConditionalEntropyLoss, NTXentLoss, Coral, MMD_loss, ConditionalEntropyLoss, EMA, VAT, SinkhornDistance
import numpy as np
import torch.optim as optim
from pytorch_metric_learning import losses

class GRAPHFiller(pl.LightningModule):
    def __init__(self, 
                 model_class,
                 model_kwargs, 
                 batch_size,
                 input_size,
                 d_in,
                 d_hidden,
                 adj_matrix,
                 optim_class, 
                 optim_kwargs, 
                 loss_fn, 
                 whiten_prob = 0.05,
                 scaled_target=False,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 mix_ratio = None, 
                 temporal_shift = None, 
                 da_method = None, 
                 aux_weight = None,
                 loader_size = 100,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        super(GRAPHFiller, self).__init__()
        self.model_cls = model_class
        self.model_kwargs = model_kwargs
        self.scaled_target = scaled_target
        self.batch_size = batch_size
        self.input_size = input_size
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.adj = adj_matrix
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        self.mix_ratio = mix_ratio
        self.temporal_shift = temporal_shift
        self.da_method = da_method
        self.loss_fn = loss_fn
        self.aux_weight = aux_weight
        self.loader_size = loader_size
        if scheduler_kwargs is None:
            self.scheduler_kwargs = dict()
        else:
            self.scheduler_kwargs = scheduler_kwargs

        src_nodes, dst_nodes = torch.nonzero(torch.tensor(self.adj), as_tuple=True)
        self.edge_index = torch.stack((src_nodes, dst_nodes))

        if torch.cuda.is_available():
            self.device = device
    
        if loss_fn is not None:
            self.loss_fn = self._check_metric(loss_fn, on_step=True)
        else:
            self.loss_fn = None

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)
        # instantiate model
        self.model = self.model_cls(**self.model_kwargs)
      
        self.keep_prob = 1. - whiten_prob

        if self.da_method == 'dann':
            self.domain_classifier = Discriminator(input_features=self.d_hidden*self.d_in)

        if self.da_method == 'cdan':
            self.domain_classifier = Discriminator_CDAN(self.input_size * self.d_hidden)

        if self.da_method == 'dirt':
            self.discriminator = Discriminator(input_features=self.d_hidden*self.d_in)
            self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
        if self.da_method == 'AdvSKM':
            self.mmd_loss = MMD_loss()
            self.AdvSKM_embedder = AdvSKM_Disc(self.d_hidden, self.d_in)
            self.optimizer_disc = optim.Adam(self.AdvSKM_embedder.parameters(),lr=0.0005,weight_decay=0.0001)

        if self.da_method == 'raincoat':
            self.feature_extractor = tf_encoder(self.input_size, self.d_in)
            self.decoder = tf_decoder(self.input_size, self.d_in, 128)
            self.recons = nn.L1Loss(reduction='sum')
            self.pi = torch.acos(torch.zeros(1)).item() * 2
            self.loss_func = losses.ContrastiveLoss(pos_margin=0.5)
            self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
      
    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, MaskedMetric):
            if 'reduction' in inspect.getfullargspec(metric).args:
                metric_kwargs = {'reduction': 'none'}
            else:
                metric_kwargs = dict()
            return MaskedMetric(metric, compute_on_step=on_step, metric_kwargs=metric_kwargs)
        return deepcopy(metric)

    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            {f'train_{k}': self._check_metric(m, on_step=True) for k, m in metrics.items()})
        self.val_metrics = MetricCollection({f'val_{k}': self._check_metric(m) for k, m in metrics.items()})
        self.test_metrics = MetricCollection({f'test_{k}': self._check_metric(m) for k, m in metrics.items()})

    
    def _preprocess(self, data, batch_preprocessing):
        """
        Perform preprocessing of a given input.

        :param data: pytorch tensor of shape [batch, steps, nodes, features] to preprocess
        :param batch_preprocessing: dictionary containing preprocessing data
        :return: preprocessed data
        """
        if isinstance(data, (list, tuple)):
            return [self._preprocess(d, batch_preprocessing) for d in data]
        trend = batch_preprocessing.get('trend', 0.)
        bias = batch_preprocessing.get('bias', 0.)
        scale = batch_preprocessing.get('scale', 1.)
        return (data - trend - bias) / (scale + epsilon)

    def _postprocess(self, data, batch_preprocessing):
        """
        Perform postprocessing (inverse transform) of a given input.
        :param data: pytorch tensor of shape [batch, steps, nodes, features] to trasform
        :param batch_preprocessing: dictionary containing preprocessing data
        :return: inverse transformed data
        """
        if isinstance(data, (list, tuple)):
            return [self._postprocess(d, batch_preprocessing) for d in data]
        trend = batch_preprocessing.get('trend', 0.)
        bias = batch_preprocessing.get('bias', 0.)
        scale = batch_preprocessing.get('scale', 1.)

        return data * (scale + epsilon) + bias + trend
    
    def predict_loader(self, loader, preprocess=False, postprocess=True, return_mask=True):
        """
        Makes predictions for an input dataloader. Returns both the predictions and the predictions targets.

        :param loader: torch dataloader
        :param preprocess: whether to preprocess the data
        :param postprocess: whether to postprocess the data
        :param return_mask: whether to return the valid mask (if it exists)
        :return: trg_yrue, y_hat
        """
        targets, imputations, masks = [], [], []
        for batch in loader:
            x,y,mask = batch['x'], batch['y'], batch['mask']
           
            y_hat, _, _ = self.model(x, self.edge_index, mask)
            targets.append(y)
            imputations.append(y_hat)
            masks.append(mask)

        y = torch.cat(targets, 0)
        y_hat = torch.cat(imputations, 0)
        if return_mask:
            mask = torch.cat(masks, 0) if masks[0] is not None else None
            return y, y_hat, mask
        return y, y_hat
    
    def impute_nans(self, x):
        isnan = torch.isnan(x)
        mask = ~isnan
        x[isnan] = torch.tensor(0.0, device=x.device)  # Temporarily replace NaNs with 0 for computation
        mean_vals = x.sum(dim=2, keepdim=True) / mask.sum(dim=2, keepdim=True)
        x[isnan] = mean_vals[isnan]
        return x
    
    def forward(self, x, mask):
        return self.model(x, self.edge_index, mask)
    
    def safe_loss_fn(self, outputs, targets, mask_valid):
        # Mask out invalid (NaN) values and compute loss only on valid data points
        outputs = outputs[mask_valid]
        targets = targets[mask_valid]
        return torch.nn.functional.mse_loss(outputs, targets)  # Example: MSE loss

    def temporal_mixup(self, src_xource, trg_xarget, mix_ratio, temporal_shift):
        batch, channels, nodes = src_xource.shape
        # Generate mixed samples based on the temporal shift and mix ratio
        # Temporal shifting should be applied along the `nodes` dimension
        src_xource = torch.where(torch.isnan(src_xource), torch.zeros_like(src_xource), src_xource)
        trg_xarget = torch.where(torch.isnan(trg_xarget), torch.zeros_like(trg_xarget), trg_xarget)
        
        # Calculate the mixing indices with temporal shifting
        indices = (torch.arange(nodes) + temporal_shift) % nodes
        
        # Perform the mixup
        src_xource_mix = mix_ratio * src_xource + (1 - mix_ratio) * trg_xarget[:, :, indices]
        trg_xarget_mix = mix_ratio * trg_xarget + (1 - mix_ratio) * src_xource[:, :, indices]

        return src_xource_mix, trg_xarget_mix
    
    def training_step(self, batch, batch_idx=0):
        loss = 0
        source_train_loader = move_data_to_device(batch['source'], self.device)
        target_train_loader = move_data_to_device(batch['target'], self.device)

        #### Unpack load
        src_x, src_y, src_mask = source_train_loader['x'], source_train_loader['y'], source_train_loader['mask'].clone().detach()
        src_pred, src_spatio_feat, src_temp_feat = self.model(src_x, self.edge_index, src_mask)
 
        trg_x, trg_y, trg_mask = target_train_loader['x'], target_train_loader['y'], target_train_loader['mask'].clone().detach()
        trg_pred, trg_spatio_feat, trg_temp_feat = self.model(trg_x, self.edge_index, trg_mask)

        #### Cross Entropy Loss
        mask_valid_s = ~torch.isnan(src_y)
        src_y_filtered = src_y.clone()
        src_pred_filtered = src_pred.clone()
        src_y_filtered[~mask_valid_s] = 0
        src_pred_filtered[~mask_valid_s] = 0
        
        loss_source = self.loss_fn(src_pred_filtered, src_y_filtered)
        
        mask_valid_t = ~torch.isnan(trg_y)
        trg_y_filtered = trg_y.clone()
        trg_pred_filtered = trg_pred.clone()
        trg_y_filtered[~mask_valid_t] = 0
        trg_pred_filtered[~mask_valid_t] = 0

        loss_target = self.loss_fn(trg_pred_filtered, trg_y_filtered)
        
        #### DA Loss
        if self.da_method != None:
            loss = loss_source + loss_target
            if self.da_method == 'direct':
                pass
            elif self.da_method == 'coral':
               
                loss_da = Coral(src_temp_feat, trg_temp_feat)

                loss_da += Coral(src_spatio_feat, trg_spatio_feat)
                loss += loss_da
            elif self.da_method == 'cotmix':
                h = self.temporal_shift // 2
                ## source dominant mix
                src_xm, trg_xm = self.temporal_mixup(src_x, trg_x, self.mix_ratio, h)
                src_ymh, _, src_ymh_r, = self.model(src_xm, self.edge_index, src_mask)
                trg_ymh, _, trg_ymh_r = self.model(trg_xm, self.edge_index, trg_mask)

                ntx = NTXentLoss(self.device, batch_size=trg_temp_feat.shape[0], temperature=0.3, use_cosine_similarity = True)
                source_ntx = ntx.forward(src_temp_feat.view(src_temp_feat.shape[0],-1), src_ymh_r.view(src_ymh_r.shape[0],-1))
                target_ntx = ntx.forward(trg_temp_feat.view(trg_temp_feat.shape[0],-1), trg_ymh_r.view(trg_ymh_r.shape[0],-1))
                loss_da = source_ntx + target_ntx
                loss = 0.2*loss_da + 0.8*loss

            elif self.da_method == 'dann':
                num_batches = self.loader_size 
                p = float(self.global_step) / (self.trainer.max_epochs * num_batches)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                domain_label_src = torch.ones(src_temp_feat.size(0), dtype=torch.long)
                domain_label_trg = torch.zeros(trg_temp_feat.size(0), dtype=torch.long)
          
                # Apply gradient reversal on source features with dynamic alpha
                src_temp_feat_reversed = ReverseLayerF.apply(src_temp_feat, alpha)
                trg_temp_feat_reversed = ReverseLayerF.apply(trg_temp_feat, alpha)

                src_spatio_feat_reversed = ReverseLayerF.apply(src_spatio_feat, alpha)
                trg_spatio_feat_reversed = ReverseLayerF.apply(trg_spatio_feat, alpha)
               
                criterion = torch.nn.CrossEntropyLoss()
                src_domain_pred1 = self.domain_classifier(src_temp_feat_reversed)
                src_domain_loss = criterion(src_domain_pred1, domain_label_src.long())

                src_domain_pred2 = self.domain_classifier(src_spatio_feat_reversed)
                src_domain_loss += criterion(src_domain_pred2, domain_label_src.long())
            
                trg_domain_pred1 = self.domain_classifier(trg_temp_feat_reversed)
                trg_domain_loss = criterion(trg_domain_pred1, domain_label_trg.long())

                trg_domain_pred2 = self.domain_classifier(trg_spatio_feat_reversed)
                trg_domain_loss += criterion(trg_domain_pred2, domain_label_trg.long())
            
                # Update the total loss with domain adaptation loss
                loss = loss + trg_domain_loss + src_domain_loss

            elif self.da_method == 'cdan':
                domain_label_src1 = torch.ones(src_temp_feat.size(0), dtype=torch.long)
                domain_label_trg1 = torch.zeros(trg_temp_feat.size(0), dtype=torch.long)

                domain_label_src2 = torch.ones(src_spatio_feat.size(0), dtype=torch.long)
                domain_label_trg2 = torch.zeros(trg_spatio_feat.size(0), dtype=torch.long)

                domain_labels1 = torch.cat([domain_label_src1, domain_label_trg1], 0)
                domain_labels2 = torch.cat([domain_label_src2, domain_label_trg2], 0)
                
                feature_concat1 = torch.cat((src_temp_feat, trg_temp_feat), dim=0)
                feature_concat2 = torch.cat((src_spatio_feat, trg_spatio_feat), dim=0)
                imputation_concat = torch.cat((src_pred, trg_pred), dim=0)

                domain_loss = 0
                self.cross_entropy = nn.CrossEntropyLoss()

                feat_x_pred1 = torch.bmm(feature_concat1, imputation_concat.transpose(1, 2)).detach()
                feat_x_pred2 = torch.bmm(feature_concat2, imputation_concat.transpose(1, 2)).detach()
                disc_prediction1 = self.domain_classifier(feat_x_pred1.view(-1, feat_x_pred1.size(1) * feat_x_pred1.size(2)))
                disc_prediction2 = self.domain_classifier(feat_x_pred1.view(-1, feat_x_pred2.size(1) * feat_x_pred2.size(2)))
           
                domain_loss = self.cross_entropy(disc_prediction1, domain_labels1)
                domain_loss += self.cross_entropy(disc_prediction2, domain_labels2)
                
                loss += domain_loss

            elif self.da_method == 'dirt':
                #### True Domain Labels
                domain_label_src = torch.ones(src_temp_feat.size(0), dtype=torch.long)
                domain_label_trg = torch.zeros(trg_temp_feat.size(0), dtype=torch.long)
          
                domain_labels = torch.cat([domain_label_src, domain_label_trg], 0)
            
                ## Feature
                feat_concat1 = torch.cat((src_temp_feat, trg_temp_feat), dim=0)
                feat_concat2 = torch.cat((src_spatio_feat, trg_spatio_feat), dim=0)
            
                self.cross_entropy = nn.CrossEntropyLoss()
                self.criterion_cond = ConditionalEntropyLoss()

                self.vat_loss = VAT(self.model, self.device)
                self.ema = EMA(0.998)
                self.ema.register(self.model)
                dis_pred1 = self.discriminator.forward(feat_concat1.detach())
                dis_pred2 = self.discriminator.forward(feat_concat2.detach())
                dis_loss = self.cross_entropy(dis_pred1, domain_labels)
                dis_loss += self.cross_entropy(dis_pred2, domain_labels)
        
                # Perform the backward pass for domain_loss explicitly
                self.optimizer_discriminator.zero_grad()
                dis_loss.backward()
                self.optimizer_discriminator.step()
               
                #### Fake Label
                domain_label_src = torch.ones(src_temp_feat.size(0), dtype=torch.long)
                domain_label_trg = torch.zeros(trg_temp_feat.size(0), dtype=torch.long)
                domain_labels = torch.cat([domain_label_src, domain_label_trg], 0)

                dis_pred1 = self.discriminator.forward(feat_concat1)
                dis_pred2 = self.discriminator.forward(feat_concat2)
                domain_loss = self.cross_entropy(dis_pred1, domain_labels)
                domain_loss += self.cross_entropy(dis_pred2, domain_labels)

                #### VADA
                loss_src_vat = self.vat_loss(src_x, src_mask, src_pred)
                loss_trg_vat = self.vat_loss(trg_x, trg_mask, trg_pred)
           
                total_vat = loss_src_vat + loss_trg_vat
                loss += total_vat + domain_loss

                self.ema(self.model)
            
            elif self.da_method == 'AdvSKM':
                source_embedding_disc = self.AdvSKM_embedder(src_temp_feat.detach())
                target_embedding_disc = self.AdvSKM_embedder(trg_temp_feat.detach())

                mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
                mmd_loss.requires_grad = True

                self.optimizer_disc.zero_grad()
                mmd_loss.backward()
                self.optimizer_disc.step()

                source_embedding_disc = self.AdvSKM_embedder(src_temp_feat)
                target_embedding_disc = self.AdvSKM_embedder(trg_temp_feat)

                mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
                mmd_loss_adv.requires_grad = True

                loss += mmd_loss_adv 
            elif self.da_method == 'raincoat':
                
                # Encode both source and target features via our time-frequency feature encoder
                src_temp_feat, out_s = self.feature_extractor.forward(src_x)   
                trg_temp_feat, out_t = self.feature_extractor.forward(trg_x)
                print(out_t.shape)
                print(trg_temp_feat.shape)
                exit()
                # Decode extracted features to time series
                src_recon = self.decoder(src_temp_feat, out_s)
                trg_recon = self.decoder(trg_temp_feat, out_t)
                # Compute reconstruction loss 
                recons = 1e-4 * (self.recons(src_recon, src_x) + self.recons(trg_recon, trg_x))
              
                # Compute alignment loss
                dr, _, _ = self.sink(src_temp_feat, trg_temp_feat)
                sink_loss = dr
              
                loss = loss + recons + sink_loss
                exit()
        else:
            loss += loss_target
        print("TRAIN LOSS:", loss.detach())
 
        return loss

    def validation_step(self, batch, dataloader_idx = 1):
        source_val_loader = move_data_to_device(batch['source'], self.device)
        target_val_loader = move_data_to_device(batch['target'], self.device)
  
        trg_yrue_list = []
        y_pred_list = []
        mask_list = []
   
        trg_x, trg_y, trg_mask = target_val_loader['x'], target_val_loader['y'], target_val_loader['mask'].clone().detach()
        trg_pred, _, _ = self.model(trg_x, self.edge_index, trg_mask)

        mask_valid_t = ~torch.isnan(trg_y)
        # print(mask_valid_t.shape)
        # exit()
        trg_y_filtered = trg_y.clone()
        trg_pred_filtered = trg_pred.clone()
        trg_y_filtered[~mask_valid_t] = 0
        trg_pred_filtered[~mask_valid_t] = 0

        trg_yrue_list.append(trg_y_filtered)
        y_pred_list.append(trg_pred_filtered)
        mask_list.append(trg_mask)

        # Concatenate predictions and ground truth values from all batches
        trg_yrue = torch.cat(trg_yrue_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        mask = torch.cat(mask_list, dim=0)

        # Calculate loss
        val_loss = self.loss_fn(y_pred, trg_yrue)

        # Update metrics
        self.val_metrics.update(y_pred, trg_yrue, mask) 
        # print("VALIDATION:", val_loss.detach())
        return val_loss

    def test_step(self, data_loader_target):
        trg_yrue_list = []
        y_pred_list = []
        mask_list = []

        for target_batch in data_loader_target:
            trg_x, trg_y, trg_mask = target_batch['x'], target_batch['y'], target_batch['mask'].clone().detach()
            trg_pred, _, _= self.model(trg_x, self.edge_index, trg_mask)

            mask_valid_t = ~torch.isnan(trg_y)
            trg_y_filtered = trg_y.clone()
            trg_pred_filtered = trg_pred.clone()
            trg_y_filtered[~mask_valid_t] = 0
            trg_pred_filtered[~mask_valid_t] = 0

            trg_yrue_list.append(trg_y_filtered)
            y_pred_list.append(trg_pred_filtered)
            mask_list.append(trg_mask)

        # Concatenate predictions and ground truth values from all batches
        trg_yrue = torch.cat(trg_yrue_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        mask = torch.cat(mask_list, dim=0)

        # Calculate loss
        test_loss = self.loss_fn(y_pred, trg_yrue)

        # Update metrics
        self.test_metrics.update(y_pred, trg_yrue, mask)  
        return test_loss

    def on_train_epoch_start(self) -> None:
        optimizers = ensure_list(self.optimizers())
        for i, optimizer in enumerate(optimizers):
            lr = optimizer.optimizer.param_groups[0]['lr']
            self.log(f'lr_{i}', lr, on_step=False, on_epoch=True, logger=True, prog_bar=False)

    def configure_optimizers(self):
        cfg = dict()
        if self.da_method == 'raincoat':
            optimizer = torch.optim.Adam(
                list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters()) + \
                list(self.parameters()),
                lr=5e-4,
                weight_decay=1e-4
            )
        else:
            optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg['optimizer'] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg['lr_scheduler'] = scheduler
            if metric is not None:
                cfg['monitor'] = metric
        return cfg
