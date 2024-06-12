import torch
import torch.nn.functional as F
import torch.nn as nn

from .gcn_layers import ResidualGatedGCNLayer, MLP


class ResidualGCNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_node = config["num_nodes"]
        self.node_dim = config["node_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.voc_nodes_in = config["voc_nodes_in"]
        self.voc_nodes_out = config["voc_nodes_out"]
        self.voc_edges_in = config["voc_edges_in"]
        self.voc_edges_out = config["voc_edges_out"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.mlp_layers = config["mlp_layers"]
        self.aggregation = config["aggregation"]
        self.nodes_coord_embedding = nn.Linear(
            self.node_dim, self.hidden_dim, bias=False
        )
        self.edges_embedding = nn.Linear(1, self.hidden_dim)

        gcn_layers = []
        for _ in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        self.mlp = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)
        # self.final_mlp = nn.Linear(self.hidden_dim, self.voc_edges_out)
        self.loss_fn = nn.NLLLoss()

    def forward(self, x_edges, x_nodes):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes, node_dim)
        Returns:
        """

        x_nodes = x_nodes.contiguous()
        x_edges = x_edges.contiguous()

        x = self.nodes_coord_embedding(x_nodes).contiguous()
        e = self.edges_embedding(x_edges.unsqueeze(-1)).contiguous()
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)
        print(x.shape)
        print(x.mean(dim=(0, 1, 2)))
        print(e.shape)
        print(e.mean(dim=(0, 1, 2)))
        e = self.mlp(e).contiguous()
        # e = self.final_mlp(e).contiguous()
        e_prob = F.softmax(e, dim=3).contiguous()
        e_log_prob = F.log_softmax(e, dim=3).contiguous()
        return e_prob, e_log_prob

    def loss(self, y_pred, y_true):
        y = y_pred.permute(
            0, 3, 1, 2
        ).contiguous()  # B x V x V x voc_edges -> B x voc_edges x V x V
        # y_true B x V x V
        print(y.shape)
        print(y_true.shape)
        loss_edges = self.loss_fn(y, y_true.contiguous())
        return loss_edges
