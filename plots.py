# plots.py


import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_entropy(probe_results, title="Accuracy and Entropy vs Layer Depth"):
    """
    Plot accuracy and entropy for 1-10 layer models
    """
    num_models = len(probe_results)
    cols = 5
    rows = (num_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    axes = axes.flatten()

    for i, num_layers in enumerate(probe_results.keys()):
        ax = axes[i]

        accuracies = [r['accuracy'] for r in probe_results[num_layers]]
        entropies = [r['entropy'] for r in probe_results[num_layers]]

        ax2 = ax.twinx()
        line1, = ax.plot(range(len(accuracies)), accuracies, 'b-o', label='Accuracy')
        line2, = ax2.plot(range(len(entropies)), entropies, 'r-s', label='Entropy')

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Accuracy (%)', color='b')
        ax2.set_ylabel('Entropy', color='r')
        ax.set_title(f'{num_layers}-Layer GCN')
        ax.grid(True, alpha=0.3)

        ax.set_xticks(range(len(accuracies)))
        ax.set_xlim(-0.3, len(accuracies) - 0.7)

        ax.legend([line1, line2], ['Accuracy', 'Entropy'], loc='center right')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_correct_vs_incorrect(trace_results, title="Entropy Dynamics - Correct vs Incorrect Nodes"):
    """
    Plot entropy for correct vs incorrect nodes
    """
    num_models = len(trace_results)
    cols = 5
    rows = (num_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    axes = axes.flatten()

    for i, num_layers in enumerate(trace_results.keys()):
        ax = axes[i]
        results = trace_results[num_layers]

        layers = [r['layer'] for r in results]
        entropy_correct = [r['entropy_final_correct'] for r in results]
        entropy_incorrect = [r['entropy_final_incorrect'] for r in results]

        ax.plot(layers, entropy_correct, 'g-o', label='Finally Correct')
        ax.plot(layers, entropy_incorrect, 'r-o', label='Finally Incorrect')

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Entropy')
        ax.set_title(f'{num_layers}-Layer GCN')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.set_xticks(range(len(layers)))
        ax.set_xlim(-0.2, len(layers) - 0.8)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_mean_entropy_vs_loss(models, data, title="Mean Entropy vs Mean Loss"):
    """
    Plot mean H and mean -log p(a*) across layers
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    num_models = len(models)
    cols = 5
    rows = (num_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    axes = axes.flatten()

    for idx, num_layers in enumerate(models.keys()):
        model = models[num_layers]
        model.eval()

        with torch.no_grad():
            _, hidden_states = model(data['X'], return_hidden=True)

        mean_H_list = []
        mean_loss_list = []

        for k, h_k in enumerate(hidden_states):
            h_k = h_k.detach()

            probe = nn.Linear(h_k.shape[1], data['num_classes']).to(h_k.device)
            optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)

            for epoch in range(100):
                logits = probe(h_k[data['train_mask']])
                loss = F.cross_entropy(logits, data['labels'][data['train_mask']])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                probs = F.softmax(probe(h_k[data['test_mask']]), dim=1)
                
                H = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                mean_H = H.mean().item()
                
                p_correct = probs[range(len(data['labels'][data['test_mask']])), data['labels'][data['test_mask']]]
                neg_log_p = -torch.log(p_correct + 1e-8)
                mean_loss = neg_log_p.mean().item()

            mean_H_list.append(mean_H)
            mean_loss_list.append(mean_loss)

        ax = axes[idx]
        ax.plot(range(len(mean_H_list)), mean_H_list, 'b-o', label='Mean H')
        ax.plot(range(len(mean_loss_list)), mean_loss_list, 'r-s', label='Mean -log p(a*)')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Value')
        ax.set_title(f'{num_layers}-Layer GCN')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_heatmap(probe_results, title="Linear Probe"):
    """
    Plot accuracy and entropy heatmaps
    """
    import numpy as np
    
    num_models = len(probe_results)
    max_layers = max(len(probe_results[k]) for k in probe_results.keys())
    
    accuracy_matrix = np.full((num_models, max_layers), np.nan)
    entropy_matrix = np.full((num_models, max_layers), np.nan)

    for i, num_layers in enumerate(probe_results.keys()):
        accuracies = [r['accuracy'] for r in probe_results[num_layers]]
        entropies = [r['entropy'] for r in probe_results[num_layers]]

        for layer_idx, (acc, ent) in enumerate(zip(accuracies, entropies)):
            accuracy_matrix[i, layer_idx] = acc
            entropy_matrix[i, layer_idx] = ent

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im1 = axes[0].imshow(accuracy_matrix, aspect='auto', cmap='YlGn')
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Model')
    axes[0].set_yticks(range(num_models))
    axes[0].set_yticklabels([f'{i}-Layer' for i in probe_results.keys()])
    axes[0].set_title(f'{title} Accuracy')
    plt.colorbar(im1, ax=axes[0], label='Accuracy (%)')

    im2 = axes[1].imshow(entropy_matrix, aspect='auto', cmap='YlOrRd')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Model')
    axes[1].set_yticks(range(num_models))
    axes[1].set_yticklabels([f'{i}-Layer' for i in probe_results.keys()])
    axes[1].set_title(f'{title} Entropy')
    plt.colorbar(im2, ax=axes[1], label='Entropy')

    plt.tight_layout()
    plt.show()
