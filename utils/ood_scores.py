"""Per-level ELBO decomposition for OOD detection in hierarchical VAEs.

Decomposes the ELBO into RE, KL1, KL2 components per sample, enabling
the L>1 OOD score from "Hierarchical VAEs Know What They Don't Know"
(Havtorn et al., 2021). Purely post-hoc — no training changes required.
"""

import numpy as np
import torch

from utils.distributions import log_normal_diag


def decompose_elbo(model, data_loader, device, exemplars_embedding=None):
    """Compute per-sample ELBO components for a hierarchical VAE.

    Decomposes the standard ELBO = RE - KL1 - KL2 into its three
    components, enabling partial scores like L>1 = RE - KL2 for
    OOD detection.

    Args:
        model:                Trained hierarchical VAE (BaseHModel subclass).
        data_loader:          DataLoader yielding (data, target) batches
                              (test/val format — no indices).
        device:               torch.device for computation.
        exemplars_embedding:  Pre-computed prior parameters from
                              load_all_pseudo_input(). None for standard prior.

    Returns:
        dict with keys 're', 'kl1', 'kl2', each a list of per-sample floats.
        - re:  log p(x|z1,z2) — reconstruction log-likelihood (negative)
        - kl1: KL(q(z1|x,z2) || p(z1|z2)) — lower level (non-negative)
        - kl2: KL(q(z2|x) || p(z2)) — upper level (non-negative)
    """
    model.eval()
    all_re, all_kl1, all_kl2 = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            data = batch[0].to(device)
            batch_size = data.size(0)

            # Forward pass: encoder + decoder
            x_mean, x_logvar, latent_stats = model.forward(data)
            z1_q, z1_q_mean, z1_q_logvar, \
                z2_q, z2_q_mean, z2_q_logvar, \
                z1_p_mean, z1_p_logvar = latent_stats

            # RE: log p(x|z1, z2)
            re = model.reconstruction_loss(data, x_mean, x_logvar)  # (batch,)

            # KL1: KL(q(z1|x,z2) || p(z1|z2))
            log_p_z1 = log_normal_diag(
                z1_q.view(-1, model.args.z1_size),
                z1_p_mean.view(-1, model.args.z1_size),
                z1_p_logvar.view(-1, model.args.z1_size), dim=1)
            log_q_z1 = log_normal_diag(
                z1_q.view(-1, model.args.z1_size),
                z1_q_mean.view(-1, model.args.z1_size),
                z1_q_logvar.view(-1, model.args.z1_size), dim=1)
            kl1 = -(log_p_z1 - log_q_z1)  # (batch,)

            # KL2: KL(q(z2|x) || p(z2))
            # Dummy indices — LOO masking is disabled in eval mode
            dummy_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            log_p_z2 = model.log_p_z(
                z=(z2_q, dummy_indices),
                exemplars_embedding=exemplars_embedding)
            log_q_z2 = log_normal_diag(
                z2_q.view(-1, model.args.z2_size),
                z2_q_mean.view(-1, model.args.z2_size),
                z2_q_logvar.view(-1, model.args.z2_size), dim=1)
            kl2 = -(log_p_z2 - log_q_z2)  # (batch,)

            all_re.extend(re.cpu().tolist())
            all_kl1.extend(kl1.cpu().tolist())
            all_kl2.extend(kl2.cpu().tolist())

    return {'re': all_re, 'kl1': all_kl1, 'kl2': all_kl2}


def save_ood_scores(id_scores, ood_scores, path):
    """Save per-sample ELBO components to CSV.

    Args:
        id_scores:  dict with 're', 'kl1', 'kl2' lists (in-distribution).
        ood_scores: dict with 're', 'kl1', 'kl2' lists (OOD), or None.
        path:       Output CSV file path.
    """
    import csv

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'sample_idx', 're', 'kl1', 'kl2', 'elbo', 'l_above_1'])

        for label, scores in [('id', id_scores), ('ood', ood_scores)]:
            if scores is None:
                continue
            for i in range(len(scores['re'])):
                re = scores['re'][i]
                kl1 = scores['kl1'][i]
                kl2 = scores['kl2'][i]
                elbo = re - kl1 - kl2
                l_above_1 = re - kl2
                writer.writerow([label, i,
                                 f'{re:.4f}', f'{kl1:.4f}', f'{kl2:.4f}',
                                 f'{elbo:.4f}', f'{l_above_1:.4f}'])


def plot_ood_histograms(id_scores, ood_scores, path):
    """Plot overlapping histograms comparing ID vs OOD score distributions.

    Creates 5 subplots in a row: RE, KL1, KL2, ELBO, L>1.
    Each subplot overlays ID (blue) and OOD (red) histograms.

    Args:
        id_scores:  dict with 're', 'kl1', 'kl2' lists.
        ood_scores: dict with 're', 'kl1', 'kl2' lists.
        path:       Output image file path (PNG).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    id_re = np.array(id_scores['re'])
    id_kl1 = np.array(id_scores['kl1'])
    id_kl2 = np.array(id_scores['kl2'])
    id_elbo = id_re - id_kl1 - id_kl2
    id_l1 = id_re - id_kl2

    ood_re = np.array(ood_scores['re'])
    ood_kl1 = np.array(ood_scores['kl1'])
    ood_kl2 = np.array(ood_scores['kl2'])
    ood_elbo = ood_re - ood_kl1 - ood_kl2
    ood_l1 = ood_re - ood_kl2

    titles = ['RE', 'KL1', 'KL2', 'ELBO', 'L>1']
    id_data = [id_re, id_kl1, id_kl2, id_elbo, id_l1]
    ood_data = [ood_re, ood_kl1, ood_kl2, ood_elbo, ood_l1]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, title, id_vals, ood_vals in zip(axes, titles, id_data, ood_data):
        ax.hist(id_vals, bins=50, alpha=0.5, color='blue', label='ID', density=True)
        ax.hist(ood_vals, bins=50, alpha=0.5, color='red', label='OOD', density=True)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
