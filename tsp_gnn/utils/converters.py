import torch
from torch_geometric.data import Data


def data_wmatrix(data: Data, wmatrix: torch.Tensor) -> Data:
    wmatrix = torch.tensor((data.num_nodes, data.num_nodes), dtype=torch.float)
    for i, u, v in enumerate(data.edge_index.t().tolist()):
        wmatrix[u, v] = data.edge_weight[i]
    return wmatrix
