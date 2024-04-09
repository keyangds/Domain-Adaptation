import torch
from torch import nn

class TransformerImputer(nn.Module):
    def __init__(self, d_in, d_model, nhead, num_encoder_layers, dim_feedforward=1024, dropout=0.1):
        super(TransformerImputer, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.input_projection = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.output_projection = nn.Linear(d_model, d_in)

    def forward(self, x, mask):
        # Permute x to match Transformer's input shape [seq_len, batch, features]
        x = x.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1).bool()
        x = torch.where(mask, x, torch.zeros_like(x))
        # Input projection
        x_proj = self.input_projection(x)
        assert not torch.isnan(x_proj).any(), "NaN detected after input projection"
        assert not torch.isinf(x_proj).any(), "Inf detected after input projection"
     
        # Transformer Encoder
        # Note: Not using generate_square_subsequent_mask
        x_encoded = self.transformer_encoder(x_proj)
    
        # Check for NaNs after Transformer Encoder
        assert not torch.isnan(x_encoded).any(), "NaN detected in x_encoded after Transformer Encoder"
    
        # Output projection
        x_out = self.output_projection(x_encoded)
        # Permute back to original shape [batch, features, seq_len]
        x_out = x_out.permute(1, 2, 0)
        return x_out, x_encoded
    
    @staticmethod
    def add_model_specific_args(parser):
        return parser

