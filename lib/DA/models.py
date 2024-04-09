from torch import nn
from torch.autograd import Function
import torch
import math
import torch.nn.functional as F
from .loss import SinkhornDistance
from pytorch_metric_learning import losses


def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]
    

###### Used for DANN
class Discriminator(nn.Module):
    """Discriminator model for domain adaptation."""

    def __init__(self, input_features, hidden_dim=256):
        """
        Init discriminator.

        Parameters:
        - input_features: The number of input features. This will be determined dynamically in the forward method.
        - hidden_dim: The dimension of the hidden layer.
        """
        super(Discriminator, self).__init__()
        self.input_features = input_features
    
        self.hidden_dim = hidden_dim

        self.layer = nn.Sequential(
            nn.Linear(self.input_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
            # nn.LogSoftmax(dim=1) might be needed depending on your loss function
        )

    def forward(self, input):
        """Forward the discriminator."""
        input = input.reshape(input.shape[0], -1)
        out = self.layer(input)
        return out

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
#### Codes required by CDAN ##############
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Discriminator_CDAN(nn.Module):
    """Discriminator model for CDAN ."""
    def __init__(self, input_features, hidden_dim=256):
        """Init discriminator."""
        super(Discriminator_CDAN, self).__init__()
        self.restored = False
        self.input_features = input_features
        self.layer = nn.Sequential(
            nn.Linear(self.input_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
    
###### Used for advSKM
class Cosine_act(nn.Module):
    def __init__(self):
        super(Cosine_act, self).__init__()

    def forward(self, input):
        return torch.cos(input)

cos_act = Cosine_act()

class AdvSKM_Disc(nn.Module):
    """Discriminator model for source domain, adjusted for input shape [batch, channel, number of nodes]."""
    
    def __init__(self, channel, num_nodes, DSKN_disc_hid=64, disc_hid_dim=64):
        """Init discriminator."""
        super(AdvSKM_Disc, self).__init__()
        
        # Calculate the input dimension based on channel and number of nodes
        self.input_dim = channel * num_nodes
        self.hid_dim = DSKN_disc_hid
        
        self.branch_1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            cos_act,
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.Linear(self.hid_dim // 2, self.hid_dim // 2),
            nn.BatchNorm1d(self.hid_dim // 2),
            cos_act
        )
        
        self.branch_2 = nn.Sequential(
            nn.Linear(self.input_dim, disc_hid_dim),
            nn.Linear(disc_hid_dim, disc_hid_dim),
            nn.BatchNorm1d(disc_hid_dim),
            nn.ReLU(),
            nn.Linear(disc_hid_dim, disc_hid_dim // 2),
            nn.Linear(disc_hid_dim // 2, disc_hid_dim // 2),
            nn.BatchNorm1d(disc_hid_dim // 2),
            nn.ReLU())

    def forward(self, input):
        """Forward the discriminator."""
        input_flat = input.view(input.size(0), -1)
       
        out_cos = self.branch_1(input_flat)
        out_rel = self.branch_2(input_flat)
        total_out = torch.cat((out_cos, out_rel), dim=1)
        
        return total_out

###### RAINCOAT
class CNN(nn.Module):
    def __init__(self, input_channels, sequence_len):
        self.channel = input_channels
        self.width = input_channels
        self.fl = sequence_len

        mid_channels = 64  # Example value
        final_out_channels = 128  # Example value
        stride = 2  # Example value
        dropout = 0.5  # Example value
        features_len = 10  # Example value, could be dynamically determined

        self.fc0 = nn.Linear(self.channel, self.width)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, mid_channels, kernel_size=8,
                      stride=stride, bias=False, padding=4),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channels, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(features_len)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.view(x.size(0), -1)
        return x_flat
    
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        # Adjustment: Ensure the weight dimensionality matches the expected FFT output
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim=-2, norm='ortho')  # Adjust FFT to operate along the sequence_length dimension
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        
        # Separate into magnitude and phase
        r = out_ft.abs()
        p = out_ft.angle()
        # Assuming the intention to concatenate magnitude and phase across the last dimension
        result = torch.cat((r, p), dim=-1)
        return result, out_ft
    
class tf_encoder(nn.Module):
    def __init__(self, input_channels, sequence_len):
        super(tf_encoder, self).__init__()
        self.modes1 = 64
        self.input_channels = input_channels
        self.sequence_len = sequence_len
        self.width = input_channels
        
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1)

        self.bn_freq = nn.BatchNorm1d(self.width * 2)  # Assuming concatenation of magnitude and phase doubles the channel size
        
        # Assuming avg is intended to reduce along the sensor dimension
        self.avg = nn.Conv1d(self.width * 2, 1, kernel_size=3, stride=1, bias=False, padding=1)


    def forward(self, x):
        ef, out_ft = self.freq_feature(x)
        ef = ef.transpose(1, 2)  # Transpose for Conv1d
        ef = self.avg(ef).squeeze()
        ef = F.relu(self.bn_freq(ef))
        et = self.cnn(x)  # Assumes CNN is dynamically adapted elsewhere
        f = torch.concat([ef, et], -1)
        return F.normalize(f), out_ft

class tf_decoder(nn.Module):
    def __init__(self, input_channels, sequence_len, final_out_channels):
        super().__init__()
        self.modes = 64
        self.bn1 = nn.BatchNorm1d(input_channels)
        self.bn2 = nn.BatchNorm1d(input_channels)
        self.convT = nn.ConvTranspose1d(final_out_channels, sequence_len, input_channels, stride=1)

    def forward(self, f, out_ft, input_channels, sequence_len, final_out_channels):
        if not hasattr(self, 'bn1'):
            self.build_layers(input_channels, sequence_len, final_out_channels)
        x_low = self.bn1(torch.fft.irfft(out_ft, n=128))
        et = f[:, self.modes * 2:]
        x_high = F.relu(self.bn2(self.convT(et.unsqueeze(2)).permute(0, 2, 1)))
        return x_low + x_high


class RAINCOAT(nn.Module):
    def __init__(self, input_channels, sequence_len, final_out_channels):
        self.feature_extractor = tf_encoder(input_channels, sequence_len)
        self.decoder = tf_decoder(input_channels, sequence_len, final_out_channels)
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters()),
            lr=5e-4,
            weight_decay=1e-4
        )
        self.coptimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters())+list(self.decoder.parameters()),
            lr=1*5e-4,
            weight_decay=1e-4
        )
            
        self.recons = nn.L1Loss(reduction='sum')
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.loss_func = losses.ContrastiveLoss(pos_margin=0.5)
        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        
    def update(self, src_x, src_y, trg_x):
  
        self.optimizer.zero_grad()
        # Encode both source and target features via our time-frequency feature encoder
        src_feat, out_s = self.feature_extractor(src_x)   
        trg_feat, out_t = self.feature_extractor(trg_x)
        # Decode extracted features to time series
        src_recon = self.decoder(src_feat, out_s)
        trg_recon = self.decoder(trg_feat, out_t)
        # Compute reconstruction loss 
        recons = 1e-4 * (self.recons(src_recon, src_x) + self.recons(trg_recon, trg_x))
        recons.backward(retain_graph=True)
        # Compute alignment loss
        dr, _, _ = self.sink(src_feat, trg_feat)
        sink_loss = dr
        sink_loss.backward(retain_graph=True)
        self.optimizer.step()
        return {'Sink': sink_loss.item()}
    
    def correct(self,src_x, src_y, trg_x):
        self.coptimizer.zero_grad()
        src_feat, out_s = self.feature_extractor(src_x)
        trg_feat, out_t = self.feature_extractor(trg_x)
        src_recon = self.decoder(src_feat, out_s)
        trg_recon = self.decoder(trg_feat, out_t)
        recons = 1e-4 * (self.recons(trg_recon, trg_x) + self.recons(src_recon, src_x))
        recons.backward()
        self.coptimizer.step()
        return {'recon': recons.item()}