import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
    
class SpatialAttention(nn.Module):
    def __init__(self, in_features, num_nodes):
        super(SpatialAttention, self).__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        # self.value = nn.Linear(in_features, in_features)
        self.softmax = nn.Softmax(dim=-1)
        self.num_nodes = num_nodes

    def forward(self, x):
        # Assuming x has shape [num_nodes, in_features]
        q = self.query(x)  
        k = self.key(x)  
        v = x

        # Correct the matrix multiplication logic
        attention_scores = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.num_nodes)  # Transpose for dot product
        attention = self.softmax(attention_scores)  # [num_nodes, num_nodes]
        attended_values = torch.matmul(attention, v)  # [num_nodes, in_features]

        return attended_values

class TemporalAttention(nn.Module):
    def __init__(self, in_features, sequence_length):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        # self.value = nn.Linear(in_features, in_features)
        self.softmax = nn.Softmax(dim=-1)
        self.sequence_length = sequence_length

    def forward(self, x):
        q = self.query(x)
        k = self.key(x).transpose(1, 2)
        v = x
        
        attention_scores = torch.matmul(q, k) / np.sqrt(self.sequence_length)
        attention = self.softmax(attention_scores)
        attended_values = torch.matmul(attention, v)
        return attended_values

class SpatioTemporalImputer(nn.Module):
    
    def __init__(self, input_size, output_size, num_nodes, num_gnn_channels, num_channels, rnn_kernel_size, dropout=0.2, sequence_length=10):
        super(SpatioTemporalImputer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_nodes = num_nodes
        self.num_gnn_channels = num_gnn_channels
        self.num_channels = num_channels
        self.rnn_kernel_size = rnn_kernel_size
        self.sequence_length = sequence_length

        self.gnn1 = GCNConv(input_size, num_gnn_channels[1], improved=True, cached=True)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_channels,
                          num_layers=rnn_kernel_size, batch_first=True, dropout=dropout if rnn_kernel_size > 1 else 0)
        self.fc_combine1 = nn.Linear(num_channels * num_nodes*2, num_channels * num_nodes // 4)
        self.fc_combine2 = nn.Linear(num_channels * num_nodes // 4, output_size * num_nodes)
        self.dropout = nn.Dropout(dropout)
        self.spatial_attention = SpatialAttention(num_gnn_channels[1], num_nodes)
        self.temporal_attention = TemporalAttention(num_channels, sequence_length)

    def forward(self, data, edge_index, masks):
        batch_size = data.size(0)
        spatial_embeddings = []
        spatial_attention = []

        data = torch.where(masks, data, torch.zeros_like(data))

        for i in range(batch_size):
            x = data[i].transpose(0, 1)
            x = torch.relu(self.gnn1(x, edge_index))
            x_a = self.spatial_attention(x)
            x = x.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
            spatial_embeddings.append(x)
            spatial_attention.append(x_a)

        spatial_embeddings = torch.stack(spatial_embeddings)
        spatial_embeddings = self.dropout(spatial_embeddings.view(batch_size, -1, self.num_gnn_channels[1]))

        spatial_attention = torch.stack(spatial_attention)
        spatial_attention = self.dropout(spatial_attention.view(batch_size, -1, self.num_gnn_channels[1]))

        # Process the original data through RNN
        rnn_input = data.view(batch_size, -1, self.input_size)  # Reshape data to fit RNN input requirements
        rnn_out, _ = self.rnn(rnn_input)  # Process through RNN
        rnn_a = self.temporal_attention(rnn_out)

        embed = torch.cat((rnn_out, spatial_embeddings), dim = -1)
        embed = embed.reshape(batch_size, -1)
        
        inter_output = self.fc_combine1(embed)
        final_output = self.fc_combine2(inter_output)
        final_output = final_output.view(batch_size, self.num_nodes, -1).transpose(1, 2)  # Reshape to match output dimensions
        
        spatial_embeddings = spatial_embeddings.transpose(1,2)
        rnn_a = rnn_a.transpose(1,2)
    
        return final_output, spatial_attention, rnn_a
    
    @staticmethod
    def add_model_specific_args(parser):
        return parser