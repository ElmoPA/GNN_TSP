import torch
import torch.nn.functional as F
import torch.nn as nn

from .gcn_layers import ResidualGatedGCNLayer, MLP


class ResidualGCNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_node = config["num_node"]
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

        self.mlp = MLP(self.hidden_dim, self.num_node, self.mlp_layers)
        self.loss_fn = nn.NLLLoss()

    def forward(self, x_edges, x_nodes):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes)
        Returns:
        """

        x = self.nodes_coord_embedding(x_nodes)
        e = self.edges_embedding(x_edges.unsqueeze(-1))
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)

        e = self.mlp(e)
        return e

    def loss(self, y_pred, y_true):
        y = F.log_softmax(y_pred, dim=3)
        y = y.permute(0, 3, 1, 2)
        loss_edges = self.loss_fn(y, y_true)
        return loss_edges


if __name__ == "__main__":
    config = {
        "num_node": 10,
        "node_dim": 2,
        "hidden_dim": 64,
        "voc_nodes_in": 10,
        "voc_nodes_out": 10,
        "voc_edges_in": 45,
        "voc_edges_out": 45,
        "num_layers": 3,
        "mlp_layers": 2,
        "aggregation": "mean",
    }
    model = ResidualGCNModel(config)
