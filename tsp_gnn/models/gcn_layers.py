import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, hidden_dim):
        super(BatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x_trans = x.transpose(1, 2).contiguous()
        x_trans_bn = self.batch_norm(x_trans)
        # print(x_trans_bn.mean(dim=1))
        # print(x_trans_bn.std(dim=1))
        x_bn = x_trans_bn.transpose(1, 2).contiguous()
        return x_bn


class BatchNormEdge(nn.Module):
    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        e_trans = e.transpose(1, 3).contiguous()  # [B, N, N, H] -> [B, H, N, N]
        e_trans_bn = self.batch_norm(e_trans)
        print(f"edge mean:{e_trans_bn.mean(dim=(0, 2, 3))}")
        print(f"edge std: {e_trans_bn.std(dim=(0, 2, 3))}")
        e_bn = e_trans_bn.transpose(1, 3).contiguous()  # [B, H, N, N] -> [B, N, N, H]
        return e_bn


class NodeFeatures(nn.Module):
    def __init__(self, hidden_dim, aggregation: str = "mean"):
        super().__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x, edge_gate):
        x = x.contiguous()
        edge_gate = edge_gate.contiguous()

        Ux = self.U(x)  # [B, N, H]
        Vx = self.V(x).contiguous()  # [B, N, H]
        Vx = Vx.unsqueeze(1).contiguous()  # [B, 1, N, H]
        gateVx = (edge_gate * Vx).contiguous()  # [B, N, N, H]

        x_new = None
        if self.aggregation == "mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (
                1e-20 + torch.sum(edge_gate, dim=2)
            )
        elif self.aggregation == "sum":
            x_new = Ux + torch.sum(gateVx, dim=2)

        return x_new.contiguous()


class EdgeFeatures(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x, e):
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)  #  [B, 1, N, H]
        Vx = Vx.unsqueeze(2)  # [B, N, 1, H]
        e_new = Ue + Wx + Vx
        return e_new


class ResidualGatedGCNLayer(nn.Module):
    def __init__(self, hidden_dim, aggregation="sum"):
        super().__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNorm(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        x = x.contiguous()
        e = e.contiguous()

        e_in = e
        x_in = x

        # edge convolution
        e_tmp = self.edge_feat(x, e)  # B x N x N x H
        e_tmp = e_tmp.contiguous()  # Ensure contiguity

        # compute edges gate
        edge_gate = F.sigmoid(e_tmp)
        x_tmp = self.node_feat(x, edge_gate)
        x_tmp = x_tmp.contiguous()  # Ensure contiguity

        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)

        e = F.relu(e_tmp).contiguous()  # Ensure contiguity
        x = F.relu(x_tmp).contiguous()  # Ensure contiguity

        x_new = x_in + x
        e_new = e_in + e

        return x_new.contiguous(), e_new.contiguous()  # Ensure output is contiguous


class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, L=2):
        super().__init__()
        self.L = L
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(2 * L)]
        )
        self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            # print(f"Layer {i}: {x[0, :, :, 0]}")
            x = F.relu(layer(x))
        return x
