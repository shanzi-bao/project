import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_probing(model, x, y_true, train_mask, test_mask, num_classes):
    model.eval()
    with torch.no_grad():
        _, hidden_states = model(x, return_hidden=True)

    results = []

    for k, h_k in enumerate(hidden_states):
        h_k = h_k.detach()

        # Create probe
        probe = nn.Linear(h_k.shape[1], num_classes).to(h_k.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)

        # Train probe
        for epoch in range(100):
            logits = probe(h_k[train_mask])
            loss = F.cross_entropy(logits, y_true[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate probe
        with torch.no_grad():
            probs = F.softmax(probe(h_k[test_mask]), dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            preds = probs.argmax(dim=1)
            accuracy = (preds == y_true[test_mask]).float().mean().item() * 100

            # Mean negative log-probability of the true class
            p_correct = probs[range(len(y_true[test_mask])), y_true[test_mask]]
            mean_neg_log_p = -torch.log(p_correct + 1e-8).mean().item()

        results.append({
            'layer': k,
            'entropy': entropy.item(),
            'accuracy': accuracy,
            'mean_neg_log_p': mean_neg_log_p
        })
        print(f"Layer {k}: entropy = {entropy:.4f}, accuracy = {accuracy:.2f}%, -log p(a*) = {mean_neg_log_p:.4f}")

    return results


def linear_probing_trace_final(model, x, y_true, train_mask, test_mask, num_classes):
    model.eval()
    with torch.no_grad():
        _, hidden_states = model(x, return_hidden=True)

    # First, train all probes and store them
    probes = []
    for k, h_k in enumerate(hidden_states):
        h_k = h_k.detach()
        probe = nn.Linear(h_k.shape[1], num_classes).to(h_k.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)

        for epoch in range(100):
            logits = probe(h_k[train_mask])
            loss = F.cross_entropy(logits, y_true[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probes.append(probe)

    # Determine correct/incorrect based on FINAL layer
    final_h = hidden_states[-1].detach()
    with torch.no_grad():
        final_probs = F.softmax(probes[-1](final_h[test_mask]), dim=1)
        final_preds = final_probs.argmax(dim=1)
        final_correct_mask = (final_preds == y_true[test_mask])
        final_incorrect_mask = ~final_correct_mask

    results = []

    # Now trace entropy for correct/incorrect nodes across ALL layers
    for k, h_k in enumerate(hidden_states):
        h_k = h_k.detach()

        with torch.no_grad():
            probs = F.softmax(probes[k](h_k[test_mask]), dim=1)

            # Per-node entropy
            node_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

            # Entropy for nodes that are FINALLY correct
            entropy_correct = node_entropy[final_correct_mask].mean().item()

            # Entropy for nodes that are FINALLY incorrect
            entropy_incorrect = node_entropy[final_incorrect_mask].mean().item()

            # Overall metrics
            preds = probs.argmax(dim=1)
            accuracy = (preds == y_true[test_mask]).float().mean().item() * 100

        results.append({
            'layer': k,
            'accuracy': accuracy,
            'entropy_overall': node_entropy.mean().item(),
            'entropy_final_correct': entropy_correct,
            'entropy_final_incorrect': entropy_incorrect,
        })

        print(f"Layer {k}: acc = {accuracy:.2f}%, "
              f"entropy(final_correct) = {entropy_correct:.4f}, "
              f"entropy(final_incorrect) = {entropy_incorrect:.4f}")

    return results



    # 加到 linear_probe.py 末尾

def linear_probing_auc(model, x, y_true, train_mask, test_mask, num_classes):
    """
    Like linear_probing_trace_final but stores per-node entropy
    and correct/incorrect masks for AUC computation.
    """
    model.eval()
    with torch.no_grad():
        _, hidden_states = model(x, return_hidden=True)

    # Train all probes
    probes = []
    for k, h_k in enumerate(hidden_states):
        h_k = h_k.detach()
        probe = nn.Linear(h_k.shape[1], num_classes).to(h_k.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)

        for epoch in range(100):
            logits = probe(h_k[train_mask])
            loss = F.cross_entropy(logits, y_true[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probes.append(probe)

    # Determine correct/incorrect based on FINAL layer
    final_h = hidden_states[-1].detach()
    with torch.no_grad():
        final_probs = F.softmax(probes[-1](final_h[test_mask]), dim=1)
        final_preds = final_probs.argmax(dim=1)
        final_correct_mask = (final_preds == y_true[test_mask]).cpu().numpy()

    results = []

    for k, h_k in enumerate(hidden_states):
        h_k = h_k.detach()

        with torch.no_grad():
            probs = F.softmax(probes[k](h_k[test_mask]), dim=1)
            node_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        results.append({
            'layer': k,
            'entropy_per_node': node_entropy.cpu().numpy(),
            'final_correct_mask': final_correct_mask,
        })

        h_c = node_entropy.cpu().numpy()[final_correct_mask].mean()
        h_i = node_entropy.cpu().numpy()[~final_correct_mask].mean()
        print(f"Layer {k}: H(correct)={h_c:.4f}, H(incorrect)={h_i:.4f}")

    return results