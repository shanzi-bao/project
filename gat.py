import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Module, ModuleList

NUM_EPOCHS = 100
LR = 0.01


def update_stats(training_stats, epoch_stats):
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key, val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats


class GATLayer(Module):
    def __init__(self, input_dim, output_dim, A, num_heads=1, concat=True):
        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat = concat
        
        A_tilde = (A > 0).to(dtype=torch.float32, device=A.device)
        A_tilde = A_tilde + torch.eye(A.size(0), device=A.device)
        self.register_buffer('adj', A_tilde)
        
        self.W = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False) for _ in range(num_heads)
        ])
        
        self.a = nn.ParameterList([
            nn.Parameter(torch.zeros(2 * output_dim, 1)) for _ in range(num_heads)
        ])
        
        for a in self.a:
            nn.init.xavier_uniform_(a.view(1, -1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        outputs = []
        
        for head in range(self.num_heads):
            h = self.W[head](x)
            N = h.size(0)
            
            a_input_i = h.unsqueeze(1).repeat(1, N, 1)
            a_input_j = h.unsqueeze(0).repeat(N, 1, 1)
            a_input = torch.cat([a_input_i, a_input_j], dim=2)
            
            e = self.leaky_relu(torch.matmul(a_input, self.a[head]).squeeze(-1))
            
            zero_vec = -1e9 * torch.ones_like(e)
            attention = torch.where(self.adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            
            h_prime = torch.matmul(attention, h)
            outputs.append(h_prime)
        
        if self.concat:
            return torch.cat(outputs, dim=1)
        else:
            return torch.mean(torch.stack(outputs), dim=0)


class SimpleGAT(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gat_layers, A, num_heads=4):
        super(SimpleGAT, self).__init__()
        self.num_gat_layers = num_gat_layers
        
        if num_gat_layers > 1:
            self.gat_layers = [GATLayer(input_dim, hidden_dim, A, num_heads=num_heads, concat=True)]
            
            for _ in range(num_gat_layers - 2):
                self.gat_layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, A, num_heads=num_heads, concat=True))
            
            self.gat_layers.append(GATLayer(hidden_dim * num_heads, output_dim, A, num_heads=1, concat=False))
        else:
            self.gat_layers = [GATLayer(input_dim, output_dim, A, num_heads=1, concat=False)]
        
        self.gat_layers = ModuleList(self.gat_layers)
    
    def forward(self, x, return_hidden=False):
        hidden_states = [x]
        
        for i in range(self.num_gat_layers - 1):
            x = self.gat_layers[i](x)
            x = F.relu(x)
            hidden_states.append(x)
        
        x = self.gat_layers[-1](x)
        hidden_states.append(x)
        
        if return_hidden:
            return x, hidden_states
        return x


def train_gat(X, y, mask, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(X)[mask]
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimiser.step()
    return loss.detach().item()


def evaluate_gat(X, y, mask, model):
    model.eval()
    with torch.no_grad():
        y_hat = model(X)[mask]
        y_hat = y_hat.max(1)[1]
        num_correct = y_hat.eq(y).sum()
        num_total = len(y)
        accuracy = 100.0 * (num_correct / num_total)
    return accuracy.item()


def train_eval_loop_gat(model, train_x, train_y, train_mask,
                        valid_x, valid_y, valid_mask,
                        test_x, test_y, test_mask):
    optimiser = Adam(model.parameters(), lr=LR)
    training_stats = None
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_gat(train_x, train_y, train_mask, model, optimiser)
        train_acc = evaluate_gat(train_x, train_y, train_mask, model)
        valid_acc = evaluate_gat(valid_x, valid_y, valid_mask, model)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}")
        
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch': epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    
    test_acc = evaluate_gat(test_x, test_y, test_mask, model)
    print(f"Our final test accuracy for the SimpleGAT is: {test_acc:.3f}")
    return training_stats