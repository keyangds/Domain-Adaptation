import torch

from . import Filler
from ..nn.models import BRITS
from pytorch_lightning.utilities import move_data_to_device
from ..DA.loss import SupConLoss, ConditionalEntropyLoss, NTXentLoss, Coral, MMD_loss, ConditionalEntropyLoss, EMA, VAT
from ..DA.models import ReverseLayerF, Discriminator, Discriminator_CDAN, RandomLayer, AdvSKM_Disc
import numpy as np

class BRITSFiller(Filler):

    def predict_loader(self, loader, preprocess=False, postprocess=True, return_mask=True):
        """
        Makes predictions for an input dataloader. Returns both the predictions and the predictions targets.

        :param loader: torch dataloader
        :param preprocess: whether to preprocess the data
        :param postprocess: whether to postprocess the data
        :param return_mask: whether to return the valid mask (if it exists)
        :return: y_true, y_hat
        """
        targets, imputations, masks = [], [], []
        for batch in loader:
            x,y,mask = batch['x'], batch['y'], batch['mask']
           
            y_hat, _, _ = self.model(x, mask)
            targets.append(y)
            imputations.append(y_hat)
            masks.append(mask)

        y = torch.cat(targets, 0)
        y_hat = torch.cat(imputations, 0)
        if return_mask:
            mask = torch.cat(masks, 0) if masks[0] is not None else None
            return y, y_hat, mask
        return y, y_hat

    def training_step(self, batch, batch_idx):
        loss = 0
        source_train_loader = move_data_to_device(batch['source'], self.device)
        target_train_loader = move_data_to_device(batch['target'], self.device)

        #### Unpack load
        x_s, y_s, mask_s = source_train_loader['x'], source_train_loader['y'], source_train_loader['mask'].clone().detach()
        x_t, y_t, mask_t = target_train_loader['x'], target_train_loader['y'], target_train_loader['mask'].clone().detach()

        # Compute predictions and compute loss
        out_s, imputations_s, predictions_s = self.model(x_s, mask_s)
        out_t, imputations_t, predictions_t = self.model(x_t, mask_t)
        
        mask_valid_s = ~torch.isnan(y_s)
        y_s_filtered = y_s.clone()
        out_s_filtered = out_s.clone()
        y_s_filtered[~mask_valid_s] = 0
        out_s_filtered[~mask_valid_s] = 0
        loss_source = self.loss_fn(y_s_filtered, out_s_filtered) 

        loss_source += BRITS.consistency_loss(*imputations_s)

        mask_valid_t = ~torch.isnan(y_t)
        y_t_filtered = y_t.clone()
        out_t_filtered = out_t.clone()
        y_t_filtered[~mask_valid_t] = 0
        out_t_filtered[~mask_valid_t] = 0

        loss_target = self.loss_fn(y_t_filtered, out_t_filtered) 
        loss_target += BRITS.consistency_loss(*imputations_t)
       
        #### DA Loss
        if self.da_method != None:
            loss = loss_source + loss_target
            
            if self.da_method == 'direct':
                pass
            elif self.da_method == 'coral':
                z_h_fwd_s = predictions_s[1]
                z_h_bwd_s = predictions_s[4]

                feature_s = torch.cat((z_h_fwd_s, z_h_bwd_s), dim=0)

                z_h_fwd_t = predictions_t[1]
                z_h_bwd_t = predictions_t[4]

                feature_t = torch.cat((z_h_fwd_t, z_h_bwd_t), dim=0)
                loss_da = Coral(feature_s, feature_t)
                loss += loss_da

            elif self.da_method == 'cotmix':
                h = self.temporal_shift // 2
                ## source dominant mix
                x_sm, x_tm = self.temporal_mixup(x_s, x_t, self.mix_ratio, h)
                y_smh, imputations_sm, predictions_sm = self.model(x_sm, mask_s)
                y_tmh, imputations_tm, predictions_tm = self.model(x_tm, mask_t)
                
                z_h_fwd_s = predictions_s[1]
                z_h_bwd_s = predictions_s[4]

                feature_s = torch.cat((z_h_fwd_s, z_h_bwd_s), dim=0)

                z_h_fwd_t = predictions_t[1]
                z_h_bwd_t = predictions_t[4]

                feature_t = torch.cat((z_h_fwd_t, z_h_bwd_t), dim=0)

                z_h_fwd_sm = predictions_sm[1]
                z_h_bwd_sm = predictions_sm[4]

                feature_sm = torch.cat((z_h_fwd_sm, z_h_bwd_sm), dim=0)

                z_h_fwd_tm = predictions_tm[1]
                z_h_bwd_tm = predictions_tm[4]

                feature_tm = torch.cat((z_h_fwd_tm, z_h_bwd_tm), dim=0)              

                ntx = NTXentLoss(self.device, batch_size=feature_sm.shape[0], temperature=0.3, use_cosine_similarity = True)

                source_ntx = ntx.forward(feature_s.view(feature_s.shape[0],-1), feature_sm.view(feature_sm.shape[0],-1))
                target_ntx = ntx.forward(feature_t.view(feature_t.shape[0],-1), feature_tm.view(feature_tm.shape[0],-1))
                loss_da = source_ntx + target_ntx
                loss = 0.2*loss_da + 0.8*loss

            elif self.da_method == 'dann':
                num_batches = self.loader_size 
                p = float(self.global_step) / (self.trainer.max_epochs * num_batches)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                z_h_fwd_s = predictions_s[1]
                z_h_bwd_s = predictions_s[4]
                feature_s = torch.cat((z_h_fwd_s, z_h_bwd_s), dim=0)

                z_h_fwd_t = predictions_t[1]
                z_h_bwd_t = predictions_t[4]
                feature_t = torch.cat((z_h_fwd_t, z_h_bwd_t), dim=0)
                
                domain_label_src = torch.ones(feature_s.size(0), dtype=torch.long)
                domain_label_trg = torch.zeros(feature_t.size(0), dtype=torch.long)
               
                domain_labels = torch.cat([domain_label_src, domain_label_trg], 0)

                # Apply gradient reversal on source features with dynamic alpha
                src_feat_reversed = ReverseLayerF.apply(feature_s, alpha)
                trg_feat_reversed = ReverseLayerF.apply(feature_t, alpha)
                combined_features_reversed = torch.cat([src_feat_reversed, trg_feat_reversed], 0)
                
                self.discriminator = Discriminator(input_features=combined_features_reversed.shape[1]*combined_features_reversed.shape[2])
                # Pass combined features through the domain discriminator
                domain_preds = self.discriminator.forward(combined_features_reversed)

                # Calculate domain adaptation loss
                criterion = torch.nn.CrossEntropyLoss()
                domain_loss = criterion(domain_preds, domain_labels)
            
                # Update the total loss with domain adaptation loss
                loss += domain_loss
        else:
            loss += loss_target
        print("TRAIN LOSS:", loss.detach())
 
        return loss
    
    def validation_step(self, batch, dataloader_idx = 1):
        target_val_loader = move_data_to_device(batch['target'], self.device)
  
        y_true_list = []
        y_pred_list = []
        mask_list = []
   
        x_t, y_t, mask_t = target_val_loader['x'], target_val_loader['y'], target_val_loader['mask'].clone().detach()
        y_th, _, _ = self.model(x_t, mask_t)

        mask_valid_t = ~torch.isnan(y_t)
        y_t_filtered = y_t.clone()
        y_th_filtered = y_th.clone()
        y_t_filtered[~mask_valid_t] = 0
        y_th_filtered[~mask_valid_t] = 0

        y_true_list.append(y_t_filtered)
        y_pred_list.append(y_th_filtered)
        mask_list.append(mask_t)

        # Concatenate predictions and ground truth values from all batches
        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        mask = torch.cat(mask_list, dim=0)

        # Calculate loss
        val_loss = self.loss_fn(y_pred, y_true)

        # Update metrics
        self.val_metrics.update(y_pred, y_true, mask) 
        # print("VALIDATION:", val_loss.detach())
        return val_loss

    def test_step(self, data_loader_target):
        y_true_list = []
        y_pred_list = []
        mask_list = []

        for target_batch in data_loader_target:
            x_t, y_t, mask_t = target_batch['x'], target_batch['y'], target_batch['mask'].clone().detach()
            y_th, imputations_t, predictions_t = self.model(x_t, mask_t)

            mask_valid_t = ~torch.isnan(y_t)
            y_t_filtered = y_t.clone()
            y_th_filtered = y_th.clone()
            y_t_filtered[~mask_valid_t] = 0
            y_th_filtered[~mask_valid_t] = 0

            y_true_list.append(y_t_filtered)
            y_pred_list.append(y_th_filtered)
            mask_list.append(mask_t)

        # Concatenate predictions and ground truth values from all batches
        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        mask = torch.cat(mask_list, dim=0)

        # Calculate loss
        test_loss = self.loss_fn(y_pred, y_true)

        # Update metrics
        self.test_metrics.update(y_pred, y_true, mask)  
        return test_loss
