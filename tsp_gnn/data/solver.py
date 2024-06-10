import queue
import torch


def floydd_warshall(dist_matrix, num_nodes):
    adjacency_matrix = dist_matrix.clone()
    adjacency_matrix[adjacency_matrix < 1e-6] = float("inf")
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if (
                    adjacency_matrix[i, k] + adjacency_matrix[k, j]
                    < adjacency_matrix[i, j]
                ):
                    adjacency_matrix[i, j] = (
                        adjacency_matrix[i, k] + adjacency_matrix[k, j]
                    )
    return adjacency_matrix


def dijkstra_path(graph, nodes):
    edges = {}
    edge_index = graph.edge_index
    edge_weight = graph.edge_weight

    for i, j, w in zip(edge_index[0], edge_index[1], edge_weight):
        if edges.get(i.item(), None) is None:
            edges[i.item()] = []
        edges[i.item()].append((j.item(), w.item()))

    path_edges = []
    path_nodes = []
    edges_list = [
        (edge_index[0, i].item(), edge_index[1, i].item())
        for i in range(edge_index.shape[1])
    ]
    for s, e in zip(nodes, nodes[1:]):
        print(s, e)

        if (s, e) in edges_list:
            path_nodes.append(s)
            path_edges.append((s, e))
            print("Skip")
            continue
        cur_path_nodes = []
        visited = torch.zeros(graph.num_nodes, dtype=torch.bool)
        prev = [-1 for i in range(graph.num_edges)]
        distances = [float("inf") for i in range(graph.num_edges)]

        q = queue.PriorityQueue()
        q.put(s)
        distances[s] = 0
        while not q.empty():
            u = q.get()
            if visited[u]:
                continue
            visited[u] = True
            for v, w in edges[u]:
                if distances[v] > distances[u] + w:
                    distances[v] = distances[u] + w
                    prev[v] = u
                    q.put(v)
        cur_n = e
        prev_n = e
        while prev[cur_n] != -1:
            cur_n = prev[cur_n]
            path_edges.append((cur_n, prev_n))
            cur_path_nodes.insert(0, cur_n)
            prev_n = cur_n
        print(cur_path_nodes)
        path_nodes.extend(cur_path_nodes)
    path_nodes.append(e)
    return path_edges, path_nodes
