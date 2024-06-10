import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import train_test_split_edges


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv3(x, edge_index)

    def decode(self, z, edge_index):
        edge_logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(edge_logits)

    def forward(self, data):
        if hasattr(data, "train_pos_edge_index"):
            z = self.encode(data.x, data.train_pos_edge_index)
        else:
            z = self.encode(data.x, data.edge_index)
        if hasattr(data, "train_pos_edge_index"):
            z = self.decode(z, data.train_pos_edge_index)
        else:
            z = self.encode(data.x, data.edge_index)
        return z
