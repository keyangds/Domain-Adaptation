import torch
import torch.nn as nn
import torch.nn.functional as F
from ...DA.models import ReverseLayerF, Discriminator

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        # Apply convolution
        x = self.conv1d(x)
        # Remove the extra right padding
        return x[:, :, :-self.padding] if self.padding > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation=dilation)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size, dropout=dropout)]
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Assume x is of shape [batch, channels, nodes]
        return self.network(x)

class TCNImputer(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNImputer, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
  

    def forward(self, x, mask):
        # x: input tensor of shape [batch, channels, nodes]
        # mask: binary mask tensor of shape [batch, channels, nodes] indicating NaNs (0 for NaNs, 1 for valid data)
        x = torch.where(mask, x, torch.zeros_like(x))
        x_tcn = self.tcn(x)
        x_imputed = self.linear(x_tcn.transpose(1, 2)).transpose(1, 2)
        # Output has the same shape as input: [batch, channels, nodes]
        # Replace NaNs in original input with predictions from TCN
        return x_imputed, x_tcn

    @staticmethod
    def add_model_specific_args(parser):
        return parser