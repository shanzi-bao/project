# @title [RUN] Hyperparameters for GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Linear, Module, ModuleList
NUM_EPOCHS =  100 #@param {type:"integer"}
LR         = 0.01 #@param {type:"number"}

# Fill in the initialisation and forward method the GCNLayer below

def update_stats(training_stats, epoch_stats):
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key, val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats

class GCNLayer(Module):
    """Graph Convolutional Network layer from Kipf & Welling.

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

    
        # Compute A_tilde = A + I (add self-loops)
        A_tilde = A + torch.eye(A.size(0), device=A.device)
        # Compute D_tilde^{-1/2}
        D_tilde = torch.diag(A_tilde.sum(dim=1))
        D_tilde_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_tilde.diagonal()))
        # Symmetric normalization: D^{-1/2} A_tilde D^{-1/2}
        self.adj_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

        # Linear transformation
        self.linear = Linear(input_dim, output_dim)
  

    def forward(self, x):
        """Implements the forward pass for the layer

        Args:
            x (torch.Tensor): input node feature matrix
        """
     
        # H = sigma(D^{-1/2} A_tilde D^{-1/2} X W)
        x = self.adj_norm @ x
        x = self.linear(x)
  
        return x
    
    # @title [RUN] Helper functions for managing experiments, training, and evaluating models

def train_gnn_cora(X, y, mask, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(X)[mask]
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimiser.step()
    return loss.data


def evaluate_gnn_cora(X, y, mask, model):
    model.eval()
    with torch.no_grad():
        y_hat = model(X)[mask]
        y_hat = y_hat.data.max(1)[1]
        num_correct = y_hat.eq(y.data).sum()
        num_total = len(y)
        accuracy = 100.0 * (num_correct/num_total)
    return accuracy


# Training loop
def train_eval_loop_gnn_cora(model, train_x, train_y, train_mask,
                        valid_x, valid_y, valid_mask,
                        test_x, test_y, test_mask
                    ):
    optimiser = Adam(model.parameters(), lr=LR)
    training_stats = None
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_gnn_cora(train_x, train_y, train_mask, model, optimiser)
        train_acc = evaluate_gnn_cora(train_x, train_y, train_mask, model)
        valid_acc = evaluate_gnn_cora(valid_x, valid_y, valid_mask, model)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}")
        # store the loss and the accuracy for the final plot
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch':epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    # Lets look at our final test performance
    test_acc = evaluate_gnn_cora(test_x, test_y, test_mask, model)
    print(f"Our final test accuracy for the SimpleGNN is: {test_acc:.3f}")
    return training_stats

# Lets see the GCNLayer in action!

class SimpleGNN(Module):
    """A Simple GNN model using the GCNLayer for node classification

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, A):
        super(SimpleGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        # Note: if a single layer is used hidden_dim should be the same as input_dim
        if num_gcn_layers > 1:
          self.gcn_layers = [GCNLayer(input_dim, hidden_dim, A)]
          self.gcn_layers += [GCNLayer(hidden_dim, hidden_dim, A) for i in range(num_gcn_layers-2)]
          self.gcn_layers += [GCNLayer(hidden_dim, output_dim, A)]
        else:
          self.gcn_layers = [GCNLayer(input_dim, output_dim, A)]

        self.gcn_layers = ModuleList(self.gcn_layers)
        self.num_gcn_layers = num_gcn_layers

    def forward(self, x, return_hidden=False):
        hidden_states = [x]  # h^(0) = original input

        for j in range(self.num_gcn_layers-1):
            x = self.gcn_layers[j](x)
            x = F.relu(x)
            hidden_states.append(x)  # h^(1), h^(2), 

        x = self.gcn_layers[-1](x)
        hidden_states.append(x)  # h^(K)

        if return_hidden:
            return x, hidden_states
        return x
