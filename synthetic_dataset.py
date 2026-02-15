# synthetic_dataset.py

import torch
from torch_geometric.utils import to_dense_adj


def load_synthetic_graph(data_path, graph_idx=0, train_ratio=0.6, val_ratio=0.2, 
                         seed=42, device="cpu"):
    """
    Load a single graph from the GraphUniverse synthetic dataset.
    
    Args:
        data_path: path to data.pt
        graph_idx: which graph to use (0-19)
        train_ratio: fraction of nodes for training
        val_ratio: fraction of nodes for validation
        seed: random seed for split
        device: 'cuda' or 'cpu'
    
    Returns:
        dict with all data tensors (same format as load_dataset in my_datasets.py)
    """
    raw = torch.load(data_path)
    all_data = raw[0]
    slices = raw[1]
    
    # Extract single graph
    x_start = slices['x'][graph_idx].item()
    x_end = slices['x'][graph_idx + 1].item()
    
    edge_start = slices['edge_index'][graph_idx].item()
    edge_end = slices['edge_index'][graph_idx + 1].item()
    
    x = all_data['x'][x_start:x_end]
    y = all_data['community_detection'][x_start:x_end]
    edge_index = all_data['edge_index'][:, edge_start:edge_end]
    
    # Reindex edges to start from 0
    edge_index = edge_index - x_start
    
    num_nodes = x.shape[0]
    num_classes = y.unique().shape[0]
    
    # Create train/val/test split
    torch.manual_seed(seed)
    perm = torch.randperm(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm[:train_size]] = True
    valid_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True
    
    # Dense adjacency matrix
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    
    return {
        'A': A.to(device),
        'X': x.to(device),
        'labels': y.to(device),
        'train_mask': train_mask.to(device),
        'valid_mask': valid_mask.to(device),
        'test_mask': test_mask.to(device),
        'train_y': y[train_mask].to(device),
        'valid_y': y[valid_mask].to(device),
        'test_y': y[test_mask].to(device),
        'num_classes': num_classes,
        'num_nodes': num_nodes,
        'graph_idx': graph_idx,
    }


def load_synthetic_all_graphs(data_path, train_ratio=0.6, val_ratio=0.2,
                               seed=42, device="cpu"):
    """
    Load all graphs from the dataset. Returns a list of dicts.
    """
    raw = torch.load(data_path)
    num_graphs = len(raw[1]['x']) - 1
    
    graphs = []
    for i in range(num_graphs):
        g = load_synthetic_graph(data_path, graph_idx=i, 
                                  train_ratio=train_ratio, val_ratio=val_ratio,
                                  seed=seed, device=device)
        graphs.append(g)
    
    return graphs