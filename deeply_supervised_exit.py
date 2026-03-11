"""
Deeply Supervised GNN with Entropy-Based Early Exit
Based on L65 project code by Shanzi Bao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Linear, Module, ModuleList
from gcn import GCNLayer

# ============================================================
# Model
# ============================================================

class DeeplySupervisedGNN(Module):
    """GNN with a linear classification head at every layer.
    
    Training: all heads contribute to a weighted loss.
    Inference: entropy of each head's prediction is used for early exit.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, A):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers (same structure as your SimpleGNN)
        if num_layers > 1:
            self.gcn_layers = [GCNLayer(input_dim, hidden_dim, A)]
            self.gcn_layers += [GCNLayer(hidden_dim, hidden_dim, A) for _ in range(num_layers - 2)]
            self.gcn_layers += [GCNLayer(hidden_dim, output_dim, A)]
        else:
            self.gcn_layers = [GCNLayer(input_dim, output_dim, A)]
        self.gcn_layers = ModuleList(self.gcn_layers)
        
        # Linear classification heads: one per layer (independent, NOT shared)
        # Layer 0 head: input_dim -> output_dim (on raw features)
        # Layer 1 to num_layers-1 heads: hidden_dim -> output_dim
        # Layer num_layers head: output_dim -> output_dim (last layer already has output_dim)
        self.heads = ModuleList()
        self.heads.append(Linear(input_dim, output_dim))  # head for h^(0) = raw input
        for k in range(1, num_layers):
            self.heads.append(Linear(hidden_dim, output_dim))  # heads for h^(1) to h^(K-1)
        self.heads.append(Linear(output_dim, output_dim))  # head for h^(K) = final layer
    
    def forward(self, x):
        """Normal forward pass. Returns final layer logits."""
        for j in range(self.num_layers - 1):
            x = self.gcn_layers[j](x)
            x = F.relu(x)
        x = self.gcn_layers[-1](x)
        return x
    
    def forward_all_heads(self, x):
        """Forward pass returning logits from ALL heads at ALL layers.
        
        Returns:
            all_logits: list of (n x num_classes) tensors, one per layer
            hidden_states: list of (n x dim) tensors
        """
        hidden_states = [x]  # h^(0)
        all_logits = [self.heads[0](x)]  # head 0 on raw input
        
        for j in range(self.num_layers - 1):
            x = self.gcn_layers[j](x)
            x = F.relu(x)
            hidden_states.append(x)
            all_logits.append(self.heads[j + 1](x))
        
        x = self.gcn_layers[-1](x)
        hidden_states.append(x)
        all_logits.append(self.heads[self.num_layers](x))
        
        return all_logits, hidden_states
    
    def forward_with_entropy_exit(self, x, threshold):
        """Inference with entropy-based early exit.
        
        For each node: compute prediction at each layer.
        If entropy < threshold, exit and use that prediction.
        
        Args:
            x: input features (n x input_dim)
            threshold: entropy threshold for exit
            
        Returns:
            final_preds: (n,) predicted class for each node
            exit_layers: (n,) which layer each node exited at
            final_probs: (n x num_classes) prediction probabilities
        """
        n = x.size(0)
        num_classes = self.heads[0].out_features
        
        final_preds = torch.zeros(n, dtype=torch.long, device=x.device)
        final_probs = torch.zeros(n, num_classes, device=x.device)
        exit_layers = torch.full((n,), self.num_layers, dtype=torch.long, device=x.device)
        exited = torch.zeros(n, dtype=torch.bool, device=x.device)
        
        h = x
        
        for k in range(self.num_layers + 1):
            # Get logits from this layer's head
            logits = self.heads[k](h)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # (n,)
            
            # Check which nodes should exit (not already exited and entropy < threshold)
            should_exit = (~exited) & (entropy < threshold)
            
            if should_exit.any():
                final_preds[should_exit] = probs[should_exit].argmax(dim=1)
                final_probs[should_exit] = probs[should_exit]
                exit_layers[should_exit] = k
                exited[should_exit] = True
            
            # If all nodes have exited, stop early
            if exited.all():
                break
            
            # Message passing to next layer (if not the last layer)
            if k < self.num_layers:
                if k < self.num_layers - 1:
                    h = F.relu(self.gcn_layers[k](h))
                else:
                    h = self.gcn_layers[k](h)
        
        # Remaining nodes that never exited: use last layer
        if not exited.all():
            remaining = ~exited
            final_preds[remaining] = probs[remaining].argmax(dim=1)
            final_probs[remaining] = probs[remaining]
        
        return final_preds, exit_layers, final_probs


# ============================================================
# Training
# ============================================================

def train_deeply_supervised(model, X, labels, train_mask, optimizer, layer_weights=None):
    """One training step with deeply supervised loss.
    
    Args:
        model: DeeplySupervisedGNN
        X: node features
        labels: ground truth labels
        train_mask: boolean mask for training nodes
        optimizer: optimizer
        layer_weights: list of weights for each layer's loss. 
                       If None, uniform weights with last layer weighted more.
    """
    model.train()
    optimizer.zero_grad()
    
    all_logits, _ = model.forward_all_heads(X)
    num_heads = len(all_logits)
    
    if layer_weights is None:
        # Default: uniform 0.3 for intermediate, 1.0 for last
        layer_weights = [0.3] * (num_heads - 1) + [1.0]
    
    total_loss = 0.0
    for k, (logits, w) in enumerate(zip(all_logits, layer_weights)):
        loss_k = F.cross_entropy(logits[train_mask], labels[train_mask])
        total_loss += w * loss_k
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()


def evaluate_deeply_supervised(model, X, labels, mask):
    """Evaluate using the LAST layer's head (standard evaluation)."""
    model.eval()
    with torch.no_grad():
        all_logits, _ = model.forward_all_heads(X)
        last_logits = all_logits[-1]
        preds = last_logits[mask].argmax(dim=1)
        accuracy = (preds == labels[mask]).float().mean().item() * 100
    return accuracy


def evaluate_with_entropy_exit(model, X, labels, mask, threshold):
    """Evaluate using entropy-based early exit."""
    model.eval()
    with torch.no_grad():
        final_preds, exit_layers, _ = model.forward_with_entropy_exit(X, threshold)
        accuracy = (final_preds[mask] == labels[mask]).float().mean().item() * 100
        avg_exit = exit_layers[mask].float().mean().item()
    return accuracy, avg_exit


def get_per_layer_entropy(model, X, mask):
    """Get entropy at each layer for analysis."""
    model.eval()
    with torch.no_grad():
        all_logits, _ = model.forward_all_heads(X)
        layer_entropies = []
        for logits in all_logits:
            probs = F.softmax(logits[mask], dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            layer_entropies.append(entropy.mean().item())
    return layer_entropies


# ============================================================
# Full experiment
# ============================================================

def run_experiment(dataset_name='cora', num_layers=10, hidden_dim=32,
                   num_epochs=200, lr=0.01, threshold=0.5, device='cpu'):
    """Run full deeply supervised + entropy exit experiment.
    
    Args:
        dataset_name: 'cora', 'pubmed', or 'citeseer'
        num_layers: number of GNN layers
        hidden_dim: hidden dimension
        num_epochs: training epochs
        lr: learning rate
        threshold: entropy threshold for exit
        device: 'cpu' or 'cuda'
    """
    from my_datasets import load_dataset
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Layers: {num_layers} | Hidden: {hidden_dim}")
    print(f"{'='*60}")
    
    # Load data
    data = load_dataset(dataset_name, device=device)
    
    # Create model
    input_dim = data['X'].shape[1]
    output_dim = data['num_classes']
    model = DeeplySupervisedGNN(
        input_dim, hidden_dim, output_dim, num_layers, data['A']
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        loss = train_deeply_supervised(
            model, data['X'], data['labels'], data['train_mask'], optimizer
        )
        
        val_acc = evaluate_deeply_supervised(
            model, data['X'], data['labels'], data['valid_mask']
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f'/tmp/best_model_{dataset_name}.pt')
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f} val_acc={val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load(f'/tmp/best_model_{dataset_name}.pt'))
    
    # Evaluate: standard (last layer)
    test_acc_standard = evaluate_deeply_supervised(
        model, data['X'], data['labels'], data['test_mask']
    )
    
    # Evaluate: entropy exit with different thresholds
    print(f"\n--- Results ---")
    print(f"Standard (last layer): {test_acc_standard:.2f}%")
    print(f"Best val acc at epoch {best_epoch}: {best_val_acc:.2f}%")
    
    print(f"\nEntropy exit results:")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Avg Exit Layer':<15}")
    print("-" * 40)
    
    for t in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        acc, avg_exit = evaluate_with_entropy_exit(
            model, data['X'], data['labels'], data['test_mask'], threshold=t
        )
        print(f"{t:<12.1f} {acc:<12.2f} {avg_exit:<15.2f}")
    
    # Per-layer entropy analysis
    print(f"\nPer-layer mean entropy (test set):")
    entropies = get_per_layer_entropy(model, data['X'], data['test_mask'])
    for k, ent in enumerate(entropies):
        print(f"  Layer {k}: {ent:.4f}")
    
    return model, data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    run_experiment(
        dataset_name=args.dataset,
        num_layers=args.layers,
        hidden_dim=args.hidden,
        num_epochs=args.epochs,
        lr=args.lr,
        threshold=args.threshold,
        device=args.device
    )
