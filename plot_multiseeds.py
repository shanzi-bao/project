# plots_multiseed.py
# Multi-seed aggregation plotting with error bars
# Data format: all_runs_xxx[seed][num_layers] = [{'layer': k, ...}, ...]

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Internal helpers: extract mean +/- std from all_runs structure
# ============================================================

def _aggregate_probe_metric(all_runs_probe, seeds, num_layers, metric):
    """
    Extract per-layer values of a given metric across multiple seeds.
    
    all_runs_probe[seed][num_layers] = [{'layer':0, 'accuracy':..., 'entropy':...}, ...]
    
    Returns:
        layers: list of layer indices
        mean_vals: np.array
        std_vals: np.array
    """
    all_vals = []
    for seed in seeds:
        vals = [r[metric] for r in all_runs_probe[seed][num_layers]]
        all_vals.append(vals)
    
    all_vals = np.array(all_vals)  # shape: (num_seeds, num_layers+1)
    mean_vals = all_vals.mean(axis=0)
    std_vals = all_vals.std(axis=0)
    layers = list(range(all_vals.shape[1]))
    return layers, mean_vals, std_vals


def _aggregate_trace_metric(all_runs_trace, seeds, num_layers, metric):
    """
    Same as _aggregate_probe_metric but for trace_results.
    """
    all_vals = []
    for seed in seeds:
        vals = [r[metric] for r in all_runs_trace[seed][num_layers]]
        all_vals.append(vals)
    
    all_vals = np.array(all_vals)
    mean_vals = all_vals.mean(axis=0)
    std_vals = all_vals.std(axis=0)
    layers = list(range(all_vals.shape[1]))
    return layers, mean_vals, std_vals


# ============================================================
# 1. Accuracy & Entropy vs Layer Depth (with error bars)
# ============================================================

def plot_accuracy_entropy(all_runs_probe, seeds, max_layers=10,
                          title="Accuracy and Entropy vs Layer Depth",
                          model_name="GAT", save_path=None):
    """
    One subplot per model depth, showing per-layer accuracy and entropy (mean +/- std).
    
    Args:
        all_runs_probe: dict[seed][num_layers] = list of dicts
        seeds: list of seed values
        max_layers: maximum model depth
        model_name: model name for subplot titles
        save_path: file path to save figure, or None to call plt.show()
    """
    cols = 5
    rows = (max_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(22, 4.5 * rows))
    axes = axes.flatten()

    for i, num_layers in enumerate(range(1, max_layers + 1)):
        ax = axes[i]
        
        layers, acc_mean, acc_std = _aggregate_probe_metric(
            all_runs_probe, seeds, num_layers, 'accuracy')
        _, ent_mean, ent_std = _aggregate_probe_metric(
            all_runs_probe, seeds, num_layers, 'entropy')

        ax2 = ax.twinx()
        
        line1, = ax.plot(layers, acc_mean, 'b-o', markersize=5, label='Accuracy')
        ax.fill_between(layers, acc_mean - acc_std, acc_mean + acc_std,
                         color='blue', alpha=0.15)
        
        line2, = ax2.plot(layers, ent_mean, 'r-s', markersize=5, label='Entropy')
        ax2.fill_between(layers, ent_mean - ent_std, ent_mean + ent_std,
                          color='red', alpha=0.15)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Accuracy (%)', color='b')
        ax2.set_ylabel('Entropy', color='r')
        ax.set_title(f'{num_layers}-Layer {model_name}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)
        ax.set_xlim(-0.3, len(layers) - 0.7)
        ax.legend([line1, line2], ['Accuracy', 'Entropy'], loc='center right', fontsize=8)

    # Hide unused subplots
    for j in range(max_layers, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{title} ({len(seeds)} seeds)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


# ============================================================
# 2. Correct vs Incorrect Entropy (with error bars)
# ============================================================

def plot_correct_vs_incorrect(all_runs_trace, seeds, max_layers=10,
                               title="Entropy Dynamics - Correct vs Incorrect Nodes",
                               model_name="GAT", save_path=None):
    """
    One subplot per model depth, showing entropy trajectories for
    finally-correct vs finally-incorrect nodes (mean +/- std).
    
    Args:
        all_runs_trace: dict[seed][num_layers] = list of dicts (from linear_probing_trace_final)
        seeds: list of seed values
    """
    cols = 5
    rows = (max_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(22, 4.5 * rows))
    axes = axes.flatten()

    for i, num_layers in enumerate(range(1, max_layers + 1)):
        ax = axes[i]
        
        layers, ent_c_mean, ent_c_std = _aggregate_trace_metric(
            all_runs_trace, seeds, num_layers, 'entropy_final_correct')
        _, ent_i_mean, ent_i_std = _aggregate_trace_metric(
            all_runs_trace, seeds, num_layers, 'entropy_final_incorrect')

        ax.plot(layers, ent_c_mean, 'g-o', markersize=5, label='Finally Correct')
        ax.fill_between(layers, ent_c_mean - ent_c_std, ent_c_mean + ent_c_std,
                         color='green', alpha=0.15)
        
        ax.plot(layers, ent_i_mean, 'r-o', markersize=5, label='Finally Incorrect')
        ax.fill_between(layers, ent_i_mean - ent_i_std, ent_i_mean + ent_i_std,
                         color='red', alpha=0.15)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Entropy')
        ax.set_title(f'{num_layers}-Layer {model_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)
        ax.set_xlim(-0.2, len(layers) - 0.8)

    for j in range(max_layers, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{title} ({len(seeds)} seeds)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


# ============================================================
# 3. Mean Entropy vs Mean Loss (with error bars, reuses stored probe results)
# ============================================================

def plot_mean_entropy_vs_loss(all_runs_probe, seeds, max_layers=10,
                               title="Mean Entropy vs Mean Loss",
                               model_name="GAT", save_path=None):
    """
    One subplot per model depth, showing mean H and mean -log p(a*) across layers.
    Reads directly from linear_probing results (requires 'mean_neg_log_p' field).
    
    Args:
        all_runs_probe: dict[seed][num_layers] = list of dicts (with entropy and mean_neg_log_p)
        seeds: list of seed values
    """
    cols = 5
    rows = (max_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(22, 4.5 * rows))
    axes = axes.flatten()

    for idx, num_layers in enumerate(range(1, max_layers + 1)):
        layers, H_mean, H_std = _aggregate_probe_metric(
            all_runs_probe, seeds, num_layers, 'entropy')
        _, loss_mean, loss_std = _aggregate_probe_metric(
            all_runs_probe, seeds, num_layers, 'mean_neg_log_p')
        
        ax = axes[idx]
        ax.plot(layers, H_mean, 'b-o', markersize=5, label='Mean H')
        ax.fill_between(layers, H_mean - H_std, H_mean + H_std, color='blue', alpha=0.15)
        
        ax.plot(layers, loss_mean, 'r-s', markersize=5, label=r'Mean $-\log p(a^*)$')
        ax.fill_between(layers, loss_mean - loss_std, loss_mean + loss_std, color='red', alpha=0.15)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Value')
        ax.set_title(f'{num_layers}-Layer {model_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(max_layers, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{title} ({len(seeds)} seeds)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


# ============================================================
# 4. Heatmap (averaged over multiple seeds)
# ============================================================

def plot_heatmap(all_runs_probe, seeds, max_layers=10,
                 title="Linear Probe", model_name="GAT", save_path=None):
    """
    Heatmap of mean accuracy and entropy across seeds.
    Rows = model depth, columns = layer index.
    
    Args:
        all_runs_probe: dict[seed][num_layers] = list of dicts
        seeds: list of seed values
    """
    max_probe_layers = max(
        len(all_runs_probe[seeds[0]][nl]) for nl in range(1, max_layers + 1)
    )
    
    accuracy_matrix = np.full((max_layers, max_probe_layers), np.nan)
    entropy_matrix = np.full((max_layers, max_probe_layers), np.nan)

    for i, num_layers in enumerate(range(1, max_layers + 1)):
        layers, acc_mean, _ = _aggregate_probe_metric(
            all_runs_probe, seeds, num_layers, 'accuracy')
        _, ent_mean, _ = _aggregate_probe_metric(
            all_runs_probe, seeds, num_layers, 'entropy')
        
        for layer_idx in range(len(acc_mean)):
            accuracy_matrix[i, layer_idx] = acc_mean[layer_idx]
            entropy_matrix[i, layer_idx] = ent_mean[layer_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im1 = axes[0].imshow(accuracy_matrix, aspect='auto', cmap='YlGn')
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Model Depth')
    axes[0].set_yticks(range(max_layers))
    axes[0].set_yticklabels([f'{i}-Layer' for i in range(1, max_layers + 1)])
    axes[0].set_title(f'{title} Accuracy ({model_name})')
    plt.colorbar(im1, ax=axes[0], label='Accuracy (%)')

    im2 = axes[1].imshow(entropy_matrix, aspect='auto', cmap='YlOrRd')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Model Depth')
    axes[1].set_yticks(range(max_layers))
    axes[1].set_yticklabels([f'{i}-Layer' for i in range(1, max_layers + 1)])
    axes[1].set_title(f'{title} Entropy ({model_name})')
    plt.colorbar(im2, ax=axes[1], label='Entropy')

    plt.suptitle(f'Mean over {len(seeds)} seeds', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


# ============================================================
# Usage example
# ============================================================
# Assuming you already have:
#   all_runs_probe_results[seed][num_layers] = linear_probing(...)
#   all_runs_trace_results[seed][num_layers] = linear_probing_trace_final(...)
#   SEEDS = [42, 123, 456, 789, 1024]
#
# plot_accuracy_entropy(
#     all_runs_probe_results, SEEDS, max_layers=10,
#     model_name="GAT", save_path="./results/acc_entropy_gat_cora.png"
# )
#
# plot_correct_vs_incorrect(
#     all_runs_trace_results, SEEDS, max_layers=10,
#     model_name="GAT", save_path="./results/correct_incorrect_gat_cora.png"
# )
#
# plot_mean_entropy_vs_loss(
#     all_runs_probe_results, SEEDS, max_layers=10,
#     model_name="GAT", save_path="./results/H_vs_loss_gat_cora.png"
# )
#
# plot_heatmap(
#     all_runs_probe_results, SEEDS, max_layers=10,
#     model_name="GAT", save_path="./results/heatmap_gat_cora.png"
# )