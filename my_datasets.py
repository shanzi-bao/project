# datasets.py

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj


class CoraDataset(object):
    def __init__(self, split="full"):
        super(CoraDataset, self).__init__()
        cora_pyg = Planetoid(root='/tmp/Cora', name='Cora', split=split)
        self.data = cora_pyg[0]
        self.train_mask = self.data.train_mask
        self.valid_mask = self.data.val_mask
        self.test_mask = self.data.test_mask
        self.num_classes = 7

    def train_val_test_split(self):
        train_x = self.data.x[self.data.train_mask]
        train_y = self.data.y[self.data.train_mask]
        valid_x = self.data.x[self.data.val_mask]
        valid_y = self.data.y[self.data.val_mask]
        test_x = self.data.x[self.data.test_mask]
        test_y = self.data.y[self.data.test_mask]
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def get_fullx(self):
        return self.data.x

    def get_labels(self):
        return self.data.y

    def get_adjacency_matrix(self):
        return to_dense_adj(self.data.edge_index)[0]


class CiteSeerDataset(object):
    def __init__(self, split="full"):
        super(CiteSeerDataset, self).__init__()
        citeseer_pyg = Planetoid(root='/tmp/CiteSeer', name='CiteSeer', split=split)
        self.data = citeseer_pyg[0]
        self.train_mask = self.data.train_mask
        self.valid_mask = self.data.val_mask
        self.test_mask = self.data.test_mask
        self.num_classes = 6

    def train_val_test_split(self):
        train_x = self.data.x[self.data.train_mask]
        train_y = self.data.y[self.data.train_mask]
        valid_x = self.data.x[self.data.val_mask]
        valid_y = self.data.y[self.data.val_mask]
        test_x = self.data.x[self.data.test_mask]
        test_y = self.data.y[self.data.test_mask]
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def get_fullx(self):
        return self.data.x

    def get_labels(self):
        return self.data.y

    def get_adjacency_matrix(self):
        return to_dense_adj(self.data.edge_index)[0]


class PubMedDataset(object):
    def __init__(self, split="full"):
        super(PubMedDataset, self).__init__()
        pubmed_pyg = Planetoid(root='/tmp/PubMed', name='PubMed', split=split)
        self.data = pubmed_pyg[0]
        self.train_mask = self.data.train_mask
        self.valid_mask = self.data.val_mask
        self.test_mask = self.data.test_mask
        self.num_classes = 3

    def train_val_test_split(self):
        train_x = self.data.x[self.data.train_mask]
        train_y = self.data.y[self.data.train_mask]
        valid_x = self.data.x[self.data.val_mask]
        valid_y = self.data.y[self.data.val_mask]
        test_x = self.data.x[self.data.test_mask]
        test_y = self.data.y[self.data.test_mask]
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def get_fullx(self):
        return self.data.x

    def get_labels(self):
        return self.data.y

    def get_adjacency_matrix(self):
        return to_dense_adj(self.data.edge_index)[0]


def load_dataset(name, split="full", device="cpu"):
    """
    Load dataset and move to device
    
    Args:
        name: 'cora', 'citeseer', or 'pubmed'
        split: 'full' or 'public'
        device: 'cuda' or 'cpu'
    
    Returns:
        dict with all data tensors
    """
    if name.lower() == 'cora':
        dataset = CoraDataset(split=split)
    elif name.lower() == 'citeseer':
        dataset = CiteSeerDataset(split=split)
    elif name.lower() == 'pubmed':
        dataset = PubMedDataset(split=split)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_x, train_y, valid_x, valid_y, test_x, test_y = dataset.train_val_test_split()

    return {
        'A': dataset.get_adjacency_matrix().to(device),
        'X': dataset.get_fullx().to(device),
        'labels': dataset.get_labels().to(device),
        'train_mask': dataset.train_mask.to(device),
        'valid_mask': dataset.valid_mask.to(device),
        'test_mask': dataset.test_mask.to(device),
        'train_y': train_y.to(device),
        'valid_y': valid_y.to(device),
        'test_y': test_y.to(device),
        'num_classes': dataset.num_classes
    }