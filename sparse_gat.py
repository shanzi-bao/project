import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

class SparseGATLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_index, num_heads=1, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        
        self.register_buffer('edge_index', edge_index)
        
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_features))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
    
    def forward(self, h):
        N = h.size(0)
        src, dst = self.edge_index
        
        h = self.W(h).view(N, self.num_heads, self.out_features)
        
        e_src = (h[src] * self.a_src).sum(dim=-1)
        e_dst = (h[dst] * self.a_dst).sum(dim=-1)
        e = F.leaky_relu(e_src + e_dst, 0.2)
        
        e_max = torch.zeros(N, self.num_heads, device=h.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce='amax', include_self=False)
        e = torch.exp(e - e_max[dst])
        
        e_sum = torch.zeros(N, self.num_heads, device=h.device)
        e_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(e), e)
        alpha = e / (e_sum[dst] + 1e-16)
        
        msg = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros(N, self.num_heads, self.out_features, device=h.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)
        
        if self.concat:
            return out.view(N, self.num_heads * self.out_features)
        else:
            return out.mean(dim=1)


class SparseGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gat_layers, edge_index, num_heads=4):
        super().__init__()
        self.num_gat_layers = num_gat_layers
        
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=None)
        
        self.layers = nn.ModuleList()
        
        if num_gat_layers == 1:
            self.layers.append(SparseGATLayer(input_dim, output_dim, edge_index,
                                               num_heads=1, concat=False))
        else:
            self.layers.append(SparseGATLayer(input_dim, hidden_dim, edge_index,
                                               num_heads=num_heads, concat=True))
            for _ in range(num_gat_layers - 2):
                self.layers.append(SparseGATLayer(hidden_dim * num_heads, hidden_dim, edge_index,
                                                   num_heads=num_heads, concat=True))
            self.layers.append(SparseGATLayer(hidden_dim * num_heads, output_dim, edge_index,
                                               num_heads=1, concat=False))
    
    def forward(self, x, return_hidden=False):
        hidden_states = [x] if return_hidden else None
        
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.elu(h)
            if return_hidden:
                hidden_states.append(h)
        
        if return_hidden:
            return h, hidden_states
        return h