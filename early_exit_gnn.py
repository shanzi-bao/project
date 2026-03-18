"""
Deeply Supervised GNN with Entropy-Based Early Exit

Training: all layers run, each layer has a linear head, 
          total loss = sum(w_k * L_k)
Inference: check entropy at each layer, exit when H < theta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ============================================================
# GCN Layer (dense, for Cora/PubMed)
# ============================================================

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, A):
        super().__init__()
        A_tilde = A + torch.eye(A.size(0), device=A.device)
        D_tilde = torch.diag(A_tilde.sum(dim=1))
        D_tilde_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_tilde.diagonal()))
        self.adj_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(self.adj_norm @ x)


# ============================================================
# Deeply Supervised GNN with Early Exit
# ============================================================

class DeeplySupervisedGNN(nn.Module):
    """
    GNN backbone + per-layer linear classification heads.
    
    Training: forward through all layers, return all logits for deep supervision.
    Inference: check entropy at each layer, exit early if confident.
    
    Args:
        input_dim: input feature dimension
        hidden_dim: hidden layer dimension
        output_dim: number of classes
        num_layers: number of GNN layers
        A: adjacency matrix
        residual: whether to use residual connections
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, A, residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.residual = residual

        # Input projection: map input features to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers: all hidden_dim -> hidden_dim
        self.gnn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, A) for _ in range(num_layers)
        ])

        # Per-layer linear classification heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Full forward pass (used during training).
        Returns list of logits from each layer's head.
        """
        h = self.input_proj(x)
        all_logits = []

        for k in range(self.num_layers):
            h_new = self.gnn_layers[k](h)
            h_new = F.relu(h_new)
            if self.residual:
                h = h + h_new  # residual connection
            else:
                h = h_new
            all_logits.append(self.heads[k](h))

        return all_logits

    def forward_early_exit(self, x, theta, node_mask=None):
        """
        Inference with entropy-based early exit.
        
        Args:
            x: input features [N, input_dim]
            theta: entropy threshold for exit
            node_mask: optional mask for which nodes to evaluate (e.g. test nodes)
            
        Returns:
            final_preds: predicted class for each node [N] or [num_masked]
            exit_layers: which layer each node exited at [N] or [num_masked]
        """
        h = self.input_proj(x)
        N = x.size(0)

        if node_mask is not None:
            num_nodes = node_mask.sum().item()
        else:
            num_nodes = N

        # Track which nodes have exited
        exited = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        final_preds = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        final_probs = torch.zeros(num_nodes, self.output_dim, device=x.device)
        exit_layers = torch.full((num_nodes,), self.num_layers - 1, 
                                  dtype=torch.long, device=x.device)

        for k in range(self.num_layers):
            h_new = self.gnn_layers[k](h)
            h_new = F.relu(h_new)
            if self.residual:
                h = h + h_new
            else:
                h = h_new

            # Get predictions for relevant nodes
            if node_mask is not None:
                logits_k = self.heads[k](h[node_mask])
            else:
                logits_k = self.heads[k](h)

            probs_k = F.softmax(logits_k, dim=1)
            entropy_k = -(probs_k * torch.log(probs_k + 1e-8)).sum(dim=1)

            # Nodes that should exit at this layer: low entropy AND haven't exited yet
            should_exit = (entropy_k < theta) & (~exited)

            if should_exit.any():
                final_preds[should_exit] = probs_k[should_exit].argmax(dim=1)
                final_probs[should_exit] = probs_k[should_exit]
                exit_layers[should_exit] = k
                exited[should_exit] = True

            # If all nodes have exited, stop early
            if exited.all():
                break

        # Remaining nodes that never exited: use last layer's prediction
        if not exited.all():
            remaining = ~exited
            final_preds[remaining] = probs_k[remaining].argmax(dim=1)
            final_probs[remaining] = probs_k[remaining]

        return final_preds, final_probs, exit_layers


# ============================================================
# Baseline GNN (no deep supervision, standard training)
# ============================================================

class BaselineGNN(nn.Module):
    """
    Standard GNN baseline: same architecture but only the final layer
    has a classification head. No deep supervision, no early exit.
    This is the control to compare against DeeplySupervisedGNN.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, A, residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.residual = residual

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, A) for _ in range(num_layers)
        ])

        # Only ONE classification head at the final layer
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for k in range(self.num_layers):
            h_new = self.gnn_layers[k](h)
            h_new = F.relu(h_new)
            if self.residual:
                h = h + h_new
            else:
                h = h_new
        return self.classifier(h)


def train_baseline(model, x, y_true, train_mask, optimizer):
    """Standard training step for BaselineGNN."""
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits[train_mask], y_true[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_baseline(model, x, y_true, mask):
    """Evaluate BaselineGNN accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = logits[mask].argmax(dim=1)
        accuracy = (preds == y_true[mask]).float().mean().item() * 100
    return accuracy


def train_eval_loop_baseline(model, x, y_true, train_mask, val_mask, test_mask,
                              num_epochs=100, lr=0.01, weight_decay=5e-4,
                              verbose=True):
    """Full training loop for BaselineGNN."""
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        loss = train_baseline(model, x, y_true, train_mask, optimizer)
        train_acc = evaluate_baseline(model, x, y_true, train_mask)
        val_acc = evaluate_baseline(model, x, y_true, val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | loss: {loss:.4f} | "
                  f"train: {train_acc:.2f}% | val: {val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = evaluate_baseline(model, x, y_true, test_mask)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"\n  Baseline test accuracy: {test_acc:.2f}%")
        print(f"  Parameters: {num_params}")

    return {
        'test_acc': test_acc,
        'num_params': num_params,
    }


# ============================================================
# Training
# ============================================================

def train_deeply_supervised(model, x, y_true, train_mask, optimizer, 
                            layer_weights=None):
    """
    One training step with deep supervision.
    
    Args:
        model: DeeplySupervisedGNN
        x: node features
        y_true: ground truth labels
        train_mask: boolean mask for training nodes
        optimizer: optimizer
        layer_weights: weights w_k for each layer's loss (default: uniform)
    """
    model.train()
    optimizer.zero_grad()

    all_logits = model(x)  # list of [N, C] logits, one per layer
    num_layers = len(all_logits)

    if layer_weights is None:
        layer_weights = [1.0 / num_layers] * num_layers

    total_loss = 0.0
    per_layer_loss = []
    for k, logits_k in enumerate(all_logits):
        loss_k = F.cross_entropy(logits_k[train_mask], y_true[train_mask])
        total_loss += layer_weights[k] * loss_k
        per_layer_loss.append(loss_k.item())

    total_loss.backward()
    optimizer.step()

    return total_loss.item(), per_layer_loss


# ============================================================
# Evaluation
# ============================================================

def evaluate_no_exit(model, x, y_true, mask):
    """Evaluate using the LAST layer's head (no early exit, like standard GNN)."""
    model.eval()
    with torch.no_grad():
        all_logits = model(x)
        final_logits = all_logits[-1]  # last layer
        preds = final_logits[mask].argmax(dim=1)
        accuracy = (preds == y_true[mask]).float().mean().item() * 100
    return accuracy


def evaluate_per_layer(model, x, y_true, mask):
    """Evaluate each layer's head independently (diagnostic)."""
    model.eval()
    with torch.no_grad():
        all_logits = model(x)
        results = []
        for k, logits_k in enumerate(all_logits):
            probs_k = F.softmax(logits_k[mask], dim=1)
            preds_k = probs_k.argmax(dim=1)
            acc_k = (preds_k == y_true[mask]).float().mean().item() * 100
            entropy_k = -(probs_k * torch.log(probs_k + 1e-8)).sum(dim=1).mean().item()
            results.append({
                'layer': k,
                'accuracy': acc_k,
                'entropy': entropy_k,
            })
    return results


def evaluate_early_exit(model, x, y_true, mask, theta):
    """Evaluate with entropy-based early exit."""
    model.eval()
    with torch.no_grad():
        final_preds, final_probs, exit_layers = model.forward_early_exit(
            x, theta=theta, node_mask=mask
        )
        accuracy = (final_preds == y_true[mask]).float().mean().item() * 100
        avg_exit = exit_layers.float().mean().item()
    return accuracy, avg_exit, exit_layers


def find_best_alpha(model, x, y_true, val_mask, num_classes, 
                    alpha_candidates=None):
    """
    Find the best exit threshold using normalized alpha.
    
    Threshold is defined as: theta = alpha * ln(C)
    where C is the number of classes. This makes alpha comparable
    across datasets with different numbers of classes.
    
    alpha=0: exit immediately (theta=0)
    alpha=1: never exit (theta=max entropy)
    
    Args:
        num_classes: number of classes C
        alpha_candidates: list of alpha values to try
    """
    import math
    max_entropy = math.log(num_classes)
    
    if alpha_candidates is None:
        alpha_candidates = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

    best_alpha = alpha_candidates[-1]
    best_acc = 0.0
    best_theta = max_entropy

    for alpha in alpha_candidates:
        theta = alpha * max_entropy
        acc, avg_exit, _ = evaluate_early_exit(model, x, y_true, val_mask, theta)
        if acc > best_acc or (acc == best_acc and alpha < best_alpha):
            # Among equal accuracies, prefer smaller alpha (earlier exit)
            best_acc = acc
            best_alpha = alpha
            best_theta = theta

    return best_alpha, best_theta, best_acc


# ============================================================
# Full Training Loop
# ============================================================

def train_eval_loop(model, x, y_true, train_mask, val_mask, test_mask,
                    num_classes, num_epochs=100, lr=0.01, weight_decay=5e-4,
                    layer_weights=None, verbose=True):
    """
    Full training and evaluation loop.
    
    Args:
        num_classes: number of classes (needed for normalized threshold)
    
    Returns:
        results dict with training stats and final metrics
    """
    import math
    max_entropy = math.log(num_classes)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0.0
    best_state = None
    training_history = []

    for epoch in range(num_epochs):
        # Train
        total_loss, per_layer_loss = train_deeply_supervised(
            model, x, y_true, train_mask, optimizer, layer_weights
        )

        # Evaluate (no exit, last layer)
        train_acc = evaluate_no_exit(model, x, y_true, train_mask)
        val_acc = evaluate_no_exit(model, x, y_true, val_mask)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        training_history.append({
            'epoch': epoch,
            'loss': total_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        })

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | loss: {total_loss:.4f} | "
                  f"train: {train_acc:.2f}% | val: {val_acc:.2f}%")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    test_acc_no_exit = evaluate_no_exit(model, x, y_true, test_mask)

    # Per-layer diagnostics
    per_layer = evaluate_per_layer(model, x, y_true, test_mask)

    # Find best alpha on validation set
    best_alpha, best_theta, val_acc_exit = find_best_alpha(
        model, x, y_true, val_mask, num_classes
    )

    # Test with early exit
    test_acc_exit, avg_exit_layer, exit_layers = evaluate_early_exit(
        model, x, y_true, test_mask, best_theta
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results (best val model):")
        print(f"  Test accuracy (no exit, last layer): {test_acc_no_exit:.2f}%")
        print(f"  Test accuracy (early exit, α={best_alpha:.2f}, "
              f"θ=α·ln{num_classes}={best_theta:.4f}): {test_acc_exit:.2f}%")
        print(f"  Average exit layer: {avg_exit_layer:.2f} / {model.num_layers-1}")
        print(f"  Best alpha: {best_alpha}  (max entropy ln({num_classes})={max_entropy:.4f})")
        print(f"\n  Per-layer accuracy and entropy:")
        for r in per_layer:
            print(f"    Layer {r['layer']}: acc={r['accuracy']:.2f}%, "
                  f"entropy={r['entropy']:.4f} "
                  f"(={r['entropy']/max_entropy:.2f}·lnC)")

    return {
        'test_acc_no_exit': test_acc_no_exit,
        'test_acc_exit': test_acc_exit,
        'avg_exit_layer': avg_exit_layer,
        'best_alpha': best_alpha,
        'best_theta': best_theta,
        'exit_layers': exit_layers,
        'per_layer': per_layer,
        'training_history': training_history,
        'max_entropy': max_entropy,
    }
