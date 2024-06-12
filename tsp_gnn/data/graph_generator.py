import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def generate_graph(num_nodes, edge_prob=0.4):
    positions = torch.randint(0, 100, (num_nodes, 2)).float()
    adj_matrix = torch.rand(num_nodes, num_nodes) < edge_prob
    dist_matrix = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        adj_matrix[i, i] = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j]:
                dist_matrix[i, j] = torch.norm(positions[i] - positions[j])
                dist_matrix[j, i] = dist_matrix[i, j]
    edge_index, edge_weight = dense_to_sparse(dist_matrix)
    data = Data(
        x=positions, pos=positions, edge_index=edge_index, edge_weight=edge_weight
    )
    return data, dist_matrix


