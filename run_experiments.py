"""
Run deeply supervised early-exit GNN experiments on Cora and PubMed.

Usage:
    python run_experiments.py
    python run_experiments.py --dataset cora --num_layers 10
    python run_experiments.py --dataset pubmed --num_layers 20
"""

import argparse
import torch
import numpy as np
import json
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj

from early_exit_gnn import (
    DeeplySupervisedGNN, train_eval_loop,
    evaluate_per_layer, evaluate_early_exit,
    BaselineGNN, train_eval_loop_baseline
)


def load_dataset(name, device='cpu'):
    """Load Cora or PubMed via PyG Planetoid."""
    dataset = Planetoid(root=f'/tmp/{name}', name=name, split='full')
    data = dataset[0]
    A = to_dense_adj(data.edge_index)[0].to(device)
    return {
        'A': A,
        'X': data.x.to(device),
        'labels': data.y.to(device),
        'train_mask': data.train_mask.to(device),
        'val_mask': data.val_mask.to(device),
        'test_mask': data.test_mask.to(device),
        'num_classes': dataset.num_classes,
        'edge_index': data.edge_index.to(device),
    }


def run_single_experiment(dataset_name, num_layers, hidden_dim, num_epochs,
                          lr, weight_decay, residual, seed, device, verbose=True):
    """Run one experiment with a given seed: baseline + deeply supervised."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    data = load_dataset(dataset_name, device)

    # ---- Baseline GNN (no deep supervision, no early exit) ----
    if verbose:
        print(f"\n--- Baseline GNN ---")

    torch.manual_seed(seed)  # reset seed for fair comparison
    baseline_model = BaselineGNN(
        input_dim=data['X'].shape[1],
        hidden_dim=hidden_dim,
        output_dim=data['num_classes'],
        num_layers=num_layers,
        A=data['A'],
        residual=residual,
    ).to(device)

    baseline_results = train_eval_loop_baseline(
        model=baseline_model,
        x=data['X'],
        y_true=data['labels'],
        train_mask=data['train_mask'],
        val_mask=data['val_mask'],
        test_mask=data['test_mask'],
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        verbose=verbose,
    )

    # ---- Deeply Supervised GNN (with early exit) ----
    if verbose:
        print(f"\n--- Deeply Supervised + Early Exit ---")

    torch.manual_seed(seed)  # reset seed for fair comparison
    model = DeeplySupervisedGNN(
        input_dim=data['X'].shape[1],
        hidden_dim=hidden_dim,
        output_dim=data['num_classes'],
        num_layers=num_layers,
        A=data['A'],
        residual=residual,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Model: {num_layers} layers, hidden={hidden_dim}, "
              f"residual={residual}, params={num_params}")

    results = train_eval_loop(
        model=model,
        x=data['X'],
        y_true=data['labels'],
        train_mask=data['train_mask'],
        val_mask=data['val_mask'],
        test_mask=data['test_mask'],
        num_classes=data['num_classes'],
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        verbose=verbose,
    )
    results['num_params'] = num_params
    results['seed'] = seed
    results['baseline_test_acc'] = baseline_results['test_acc']
    results['baseline_num_params'] = baseline_results['num_params']
    return results


def run_multi_seed(dataset_name, num_layers, hidden_dim=32, num_epochs=100,
                   lr=0.01, weight_decay=5e-4, residual=True,
                   seeds=None, device='cpu'):
    """Run experiment across multiple seeds and aggregate."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        r = run_single_experiment(
            dataset_name, num_layers, hidden_dim, num_epochs,
            lr, weight_decay, residual, seed, device, verbose=True
        )
        all_results.append(r)

    # Aggregate
    no_exit_accs = [r['test_acc_no_exit'] for r in all_results]
    exit_accs = [r['test_acc_exit'] for r in all_results]
    avg_exits = [r['avg_exit_layer'] for r in all_results]
    baseline_accs = [r['baseline_test_acc'] for r in all_results]
    alphas = [r['best_alpha'] for r in all_results]
    max_ent = all_results[0]['max_entropy']

    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset_name}, {num_layers} layers, {len(seeds)} seeds")
    print(f"  (max entropy ln(C) = {max_ent:.4f})")
    print(f"{'='*60}")
    print(f"  Baseline accuracy:   {np.mean(baseline_accs):.2f} ± {np.std(baseline_accs):.2f}")
    print(f"  No-exit accuracy:    {np.mean(no_exit_accs):.2f} ± {np.std(no_exit_accs):.2f}")
    print(f"  Early-exit accuracy: {np.mean(exit_accs):.2f} ± {np.std(exit_accs):.2f}")
    print(f"  Avg exit layer:      {np.mean(avg_exits):.2f} ± {np.std(avg_exits):.2f}")
    print(f"  Best alpha:          {np.mean(alphas):.3f} ± {np.std(alphas):.3f}")

    # Per-layer accuracy averaged across seeds
    num_l = len(all_results[0]['per_layer'])
    print(f"\n  Per-layer test accuracy (averaged):")
    for k in range(num_l):
        accs_k = [r['per_layer'][k]['accuracy'] for r in all_results]
        ents_k = [r['per_layer'][k]['entropy'] for r in all_results]
        print(f"    Layer {k}: acc={np.mean(accs_k):.2f}±{np.std(accs_k):.2f}%, "
              f"entropy={np.mean(ents_k):.4f}±{np.std(ents_k):.4f} "
              f"({np.mean(ents_k)/max_ent:.2f}·lnC)")

    return all_results


def run_depth_comparison(dataset_name, depths=None, hidden_dim=32,
                         num_epochs=100, lr=0.01, weight_decay=5e-4,
                         seeds=None, device='cpu'):
    """
    Compare different depths to see how early exit adapts.
    This is key for showing that deeper models + early exit can match/beat
    hand-tuned shallow models.
    """
    if depths is None:
        depths = [2, 4, 6, 10, 20]
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    summary = {}
    for L in depths:
        print(f"\n{'#'*60}")
        print(f"# Depth L = {L}")
        print(f"{'#'*60}")
        results = run_multi_seed(
            dataset_name, num_layers=L, hidden_dim=hidden_dim,
            num_epochs=num_epochs, lr=lr, weight_decay=weight_decay,
            residual=True, seeds=seeds, device=device
        )
        no_exit = [r['test_acc_no_exit'] for r in results]
        exit_ = [r['test_acc_exit'] for r in results]
        avg_ex = [r['avg_exit_layer'] for r in results]
        base = [r['baseline_test_acc'] for r in results]
        alphas = [r['best_alpha'] for r in results]
        summary[L] = {
            'baseline_mean': np.mean(base),
            'baseline_std': np.std(base),
            'no_exit_mean': np.mean(no_exit),
            'no_exit_std': np.std(no_exit),
            'exit_mean': np.mean(exit_),
            'exit_std': np.std(exit_),
            'avg_exit_layer_mean': np.mean(avg_ex),
            'avg_exit_layer_std': np.std(avg_ex),
            'alpha_mean': np.mean(alphas),
            'alpha_std': np.std(alphas),
        }

    print(f"\n{'='*80}")
    print(f"DEPTH COMPARISON: {dataset_name}")
    print(f"{'='*80}")
    print(f"{'Depth':>6} | {'Baseline':>15} | {'DeepSup (last)':>15} | {'Early Exit':>15} | {'Avg Exit':>10} | {'Alpha':>10}")
    print(f"{'-'*6}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}-+-{'-'*10}")
    for L in depths:
        s = summary[L]
        print(f"{L:>6} | {s['baseline_mean']:>6.2f} ± {s['baseline_std']:<5.2f} | "
              f"{s['no_exit_mean']:>6.2f} ± {s['no_exit_std']:<5.2f} | "
              f"{s['exit_mean']:>6.2f} ± {s['exit_std']:<5.2f} | "
              f"{s['avg_exit_layer_mean']:>5.2f}±{s['avg_exit_layer_std']:<4.2f} | "
              f"{s['alpha_mean']:>5.3f}±{s['alpha_std']:<4.3f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Early-Exit GNN Experiments')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'PubMed', 'CiteSeer'])
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Single depth to test. If not set, runs depth comparison.')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--no_residual', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    if args.num_layers is not None:
        # Single depth
        run_multi_seed(
            args.dataset, args.num_layers, args.hidden_dim,
            args.num_epochs, args.lr, args.weight_decay,
            residual=not args.no_residual, seeds=args.seeds, device=device
        )
    else:
        # Depth comparison
        run_depth_comparison(
            args.dataset, depths=[2, 4, 6, 10, 20],
            hidden_dim=args.hidden_dim, num_epochs=args.num_epochs,
            lr=args.lr, weight_decay=args.weight_decay,
            seeds=args.seeds, device=device
        )


if __name__ == '__main__':
    main()
