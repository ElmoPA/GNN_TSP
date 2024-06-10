import matplotlib.pyplot as plt

import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def show_graph(graph: Data, edge_weight=True) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    g = to_networkx(graph, edge_attrs=["edge_weight"], to_undirected=False)
    nx.draw_networkx(g)


def show_graph_pos(graph: Data, edge_weight=True) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = {i: (graph.x[i].numpy()) for i in range(len(graph.x))}
    g = to_networkx(graph, edge_attrs=["edge_weight"], to_undirected=False)

    nx.draw_networkx_nodes(g, pos, ax=ax)
    nx.draw_networkx_edges(g, pos, ax=ax)

    node_labels = {
        i: f"{i}: ({graph.x[i][0].item():.1f}, {graph.x[i][1].item():.1f})"
        for i in range(len(graph.pos))
    }
    nx.draw_networkx_labels(g, pos, labels=node_labels, ax=ax)
    if edge_weight:
        edge_labels = {
            (u, v): f'{data["edge_weight"]:.2f}' for u, v, data in g.edges(data=True)
        }
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, ax=ax)

    plt.show()


def show_graph_path(graph, nodes, edges, edge_weight=True) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = {i: (graph.pos[i].numpy()) for i in range(len(graph.pos))}
    g = to_networkx(graph, edge_attrs=["edge_weight"], to_undirected=False)

    node_colors = ["red" if i in nodes else "blue" for i in range(len(graph.pos))]
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, ax=ax)

    edge_colors = [
        "red" if (u, v) in edges or (v, u) in edges else "black" for u, v in g.edges()
    ]
    nx.draw_networkx_edges(g, pos, edge_color=edge_colors, ax=ax)

    node_labels = {
        i: f"{i}: ({graph.pos[i][0].item():.1f}, {graph.pos[i][1].item():.1f})"
        for i in range(len(graph.pos))
    }
    nx.draw_networkx_labels(g, pos, labels=node_labels, ax=ax)

    if edge_weight:
        edge_labels = {
            (u, v): f'{data["edge_weight"]:.2f}' for u, v, data in g.edges(data=True)
        }
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, ax=ax)

    plt.show()
