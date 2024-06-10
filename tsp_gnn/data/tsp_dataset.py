import torch
from tqdm import tqdm
from .graph_generator import generate_graph
from .solver import floydd_warshall, dijkstra_path
from python_tsp.exact import solve_tsp_dynamic_programming


class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.dataset = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pass
        # return data, dist_matrix, label_matrix

    def generate_dataset(self, num_samples, node_prob=0.5, edge_prob=0.4):
        dataset = []
        for i in tqdm(num_samples):
            data, dist_matrix = generate_graph(self.num_nodes, edge_prob)
            tsp_nodes = torch.rand(self.num_nodes) < node_prob
            tsp_matrix = floydd_warshall(dist_matrix, self.num_nodes)[tsp_nodes][
                :, tsp_nodes
            ]
            label_matrix, distance = solve_tsp_dynamic_programming(dist_matrix)

            dataset.append((data, dist_matrix, label_matrix))

        self.dataset = dataset
