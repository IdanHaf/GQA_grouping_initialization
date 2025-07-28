import math
import numpy as np

import torch
from torch import nn
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from k_means_constrained import KMeansConstrained


def plot_grouped_heatmap(sim_matrix_np, groups, layer_idx, title):
    reordered = [h for group in groups for h in group]
    reordered_matrix = sim_matrix_np[reordered][:, reordered]

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(reordered_matrix, cmap='coolwarm', vmin=-1, vmax=1,
                     xticklabels=[f"H{i}" for i in reordered],
                     yticklabels=[f"H{i}" for i in reordered])

    group_size = len(groups[0])
    for i in range(1, len(groups)):
        plt.axhline(i * group_size, color='black', lw=1)
        plt.axvline(i * group_size, color='black', lw=1)

    # Overlay dark mask for non-diagonal blocks
    num_groups = len(groups)
    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                # Coordinates for the top-left corner of the block
                x_start = j * group_size
                y_start = i * group_size
                # Draw rectangle over non-diagonal group block
                rect = plt.Rectangle((x_start, y_start), group_size, group_size,
                                     color='black', alpha=0.4, zorder=10)
                ax.add_patch(rect)

    plt.title(f"Layer {layer_idx} - {title}")
    plt.xlabel("Head")
    plt.ylabel("Head")
    plt.tight_layout()
    plt.show()


def cluster_equal_size_on_mean_proj(proj, group_size):
    """
    :param proj: tensor of shape (n_heads, seq_len, d_q)
    :param group_size: int represents the cluster size.

    Returns: list of head-index groups, each of size `group_size`
    """
    n_heads = proj.shape[0]
    d_kv = proj.shape[2]
    n_groups = n_heads // group_size

    # Mean pool to get shape of (n_heads, d_q).
    features = proj.mean(dim=1)
    X = features.detach().cpu().numpy()

    kmeans = KMeansConstrained(
        n_clusters=n_groups,
        size_min=group_size,
        size_max=group_size,
        random_state=42
    )
    labels = kmeans.fit_predict(X)

    groups = [
        [i for i, label in enumerate(labels) if label == g]
        for g in range(n_groups)
    ]
    return groups


def equal_size_head_groups(sim_matrix_np, group_size):
    """
      :param sim_matrix_np: sim_matrix_np: (n_heads×n_heads) numpy array of avg‑cosine similarities
      :param group_size: desired # heads per cluster
      :returns - List of lists: each sublist is the head‑indices in that group
    """
    n_heads = sim_matrix_np.shape[0]
    # 1) hierarchical linkage + leaf ordering
    L = linkage(sim_matrix_np, method='ward')
    order = leaves_list(L)

    # 2) chunk into equal slices
    n_full = n_heads // group_size  # how many full groups
    groups = [
        list(order[i * group_size:(i + 1) * group_size])
        for i in range(n_full)
    ]

    # 3) optional remainder
    rem = order[n_full * group_size:]
    if rem.size > 0:
        groups.append(list(rem))  # or merge into last, whichever you prefer

    return groups


def calc_sim_matrix(proj):
    """
    Calculating the similarity matrix of the heads.

    :param proj: tensor of shape (n_heads, seq_len, d_kv)
    :return: numpy similarity matrix between the heads.
    """
    n_heads, seq_len, d_kv = proj.shape

    sim_matrix = torch.zeros((n_heads, n_heads))

    for t in range(seq_len):
        vectors = proj[:, t, :]  # (n_heads, d_kv)
        norm = F.normalize(vectors, dim=1)  # normalize each head's vector

        cos_sim = torch.matmul(norm, norm.T)  # (n_heads, n_heads)
        sim_matrix += cos_sim

    # Average across sequence length.
    sim_matrix /= seq_len
    sim_matrix_np = sim_matrix.detach().numpy()

    return sim_matrix_np


def js_divergence(attn_probs):
    """
    Calculate Jensen-Shannon divergence between attention head distributions.

    :param attn_probs: Tensor of shape (n_heads, seq_len, seq_len)
                       Softmaxed attention weights for each head.
    :return: numpy matrix of shape (n_heads, n_heads) with JSD scores.
    """
    n_heads, seq_len, _ = attn_probs.shape
    sim_matrix = torch.zeros((n_heads, n_heads))

    for i in range(n_heads):
        for j in range(n_heads):
            # Average attention distribution across all query positions
            p = attn_probs[i].mean(dim=0)  # shape: (seq_len,)
            q = attn_probs[j].mean(dim=0)

            # Normalize + add epsilon to avoid log(0)
            eps = 1e-8
            p = (p + eps) / (p + eps).sum()
            q = (q + eps) / (q + eps).sum()

            # Jensen-Shannon divergence
            m = 0.5 * (p + q)
            kl_pm = F.kl_div(p.log(), m, reduction='sum')
            kl_qm = F.kl_div(q.log(), m, reduction='sum')
            jsd = 0.5 * (kl_pm + kl_qm)

            sim_matrix[i, j] = jsd

    return sim_matrix.detach().numpy()


def plot_layer_head_similarity2(outputs, layer_idx, group_size, title):
    proj = outputs[0].squeeze(0)  # (n_heads, seq_len, seq_len)
    n_heads = proj.shape[0]

    sim_matrix_np = calc_sim_matrix(proj)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(sim_matrix_np, cmap='YlOrRd', vmin=0, vmax=0.7,  # JSD max ≈ ln(2)
                     xticklabels=[f"H{i}" for i in range(n_heads)],
                     yticklabels=[f"H{i}" for i in range(n_heads)])
    for i in range(1, (n_heads // group_size)):
        plt.axhline(i * group_size, color='black', lw=1)
        plt.axvline(i * group_size, color='black', lw=1)

    num_groups = n_heads // group_size
    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                rect = plt.Rectangle((j * group_size, i * group_size), group_size, group_size,
                                     color='black', alpha=0.4, zorder=10)
                ax.add_patch(rect)

    plt.title(f"Layer {layer_idx} - JSD between Attention Distributions")
    plt.xlabel("Head")
    plt.ylabel("Head")
    plt.tight_layout()
    plt.show()


def plot_layer_head_similarity(outputs, layer_idx, group_size, title):
    proj = outputs[0]
    proj = proj.squeeze(0)  # (n_heads, seq_len, d_kv)
    n_heads = proj.shape[0]

    sim_matrix_np = calc_sim_matrix(proj=proj)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(sim_matrix_np, cmap='coolwarm', vmin=-1, vmax=1,
                     xticklabels=[f"H{i}" for i in range(n_heads)],
                     yticklabels=[f"H{i}" for i in range(n_heads)])
    for i in range(1, (n_heads // group_size)):
        plt.axhline(i * group_size, color='black', lw=1)
        plt.axvline(i * group_size, color='black', lw=1)

    # Overlay dark mask for non-diagonal blocks
    num_groups = n_heads // group_size
    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                # Coordinates for the top-left corner of the block
                x_start = j * group_size
                y_start = i * group_size
                # Draw rectangle over non-diagonal group block
                rect = plt.Rectangle((x_start, y_start), group_size, group_size,
                                     color='black', alpha=0.4, zorder=10)
                ax.add_patch(rect)

    plt.title(f"Layer {layer_idx} - Head-to-Head Q Cosine Similarity")
    plt.xlabel("Head")
    plt.ylabel("Head")
    plt.tight_layout()
    plt.show()

    # Group and plot grouping on sim_matrix heat-map.
    grouping_via_mean_pool = cluster_equal_size_on_mean_proj(proj, group_size)
    plot_grouped_heatmap(sim_matrix_np, grouping_via_mean_pool, layer_idx=layer_idx, title=title)

    grouping_via_sim_matrix = equal_size_head_groups(sim_matrix_np, group_size)
    plot_grouped_heatmap(sim_matrix_np, grouping_via_sim_matrix, layer_idx=layer_idx, title=title)

    return grouping_via_mean_pool


def process_grouping(layer_grouping, n_heads):
    processed_grouping = np.zeros(n_heads)

    for idx, group in enumerate(layer_grouping):
        for element in group:
            processed_grouping[element] = idx

    return processed_grouping


def plot_given_grouping_on_kqv(kvq_w, n_heads, kv_heads: int, layer_idx, grouping):
    # d_model -> kv_heads*d_kv.
    proj = nn.Linear(kvq_w.shape[1], kvq_w.shape[0], bias=False)
    proj.weight.data = kvq_w
    d_kv = kvq_w.shape[0] // kv_heads

    outputs = proj(torch.rand(1, 8000, 2048))
    outputs = outputs.view(outputs.shape[0], -1, n_heads, d_kv).transpose(1, 2)
    proj = outputs[0]
    proj = proj.squeeze(0)  # (n_heads, seq_len, d_kv)

    sim_matrix_np = calc_sim_matrix(proj=proj)
    plot_grouped_heatmap(sim_matrix_np, grouping, layer_idx=layer_idx, title="K grouping on Q projection")


def close_by_w(kvq_w, n_heads, kv_heads: int, layer_idx, group_size=2, title="Clustered by Q", samples=6000):
    # d_model -> kv_heads*d_kv.
    proj = nn.Linear(kvq_w.shape[1], kvq_w.shape[0], bias=False)
    proj.weight.data = kvq_w
    d_kv = kvq_w.shape[0] // kv_heads

    outputs = proj(torch.rand(1, samples, kvq_w.shape[1]))
    outputs = outputs.view(outputs.shape[0], -1, n_heads, d_kv).transpose(1, 2)

    # Calculate cosine similarity between the heads on same input.
    grouping = plot_layer_head_similarity(outputs, layer_idx, group_size, title=title)
    grouping_processed = process_grouping(grouping, n_heads)

    return grouping_processed, grouping


def close_by_attention_score(wq, wk, n_heads, layer_idx, group_size=2, title="Clustered by attention score",
                             samples=6000):
    q_proj = nn.Linear(wq.shape[1], wq.shape[0], bias=False)
    k_proj = nn.Linear(wk.shape[1], wk.shape[0], bias=False)
    d_kv = wq.shape[0] // n_heads
    input_x = torch.rand(1, samples, wq.shape[0])
    q_outputs = q_proj(input_x)
    k_outputs = k_proj(input_x)
    q_outputs = q_outputs.view(q_outputs.shape[0], -1, n_heads, d_kv).transpose(1, 2)
    k_outputs = k_outputs.view(k_outputs.shape[0], -1, n_heads, d_kv).transpose(1, 2)
    attn_weights = torch.matmul(q_outputs, k_outputs.transpose(2, 3))
    attn_weights = attn_weights / math.sqrt(d_kv)
    attn_scores = F.softmax(attn_weights, dim=-1)

    plot_layer_head_similarity2(attn_scores, layer_idx, group_size, title=title)
