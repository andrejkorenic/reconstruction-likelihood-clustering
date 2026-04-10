from __future__ import print_function
from utils.plot_images import plot_images
from utils.plot_timeseries import (
    plot_ts_reconstruction, plot_ts_decomposition, plot_ts_generation
)
import torch
from tqdm import tqdm
from scipy.special import logsumexp
import numpy as np
import torch.nn.functional as F
import os
from pathlib import Path
import pandas as pd


def evaluate_loss(args, model, loader, dataset=None, exemplars_embedding=None):
    """Compute average ELBO, reconstruction error, and KL over a data loader.

    Args:
        args: experiment config namespace.
        model: VAE model with a calculate_loss() method.
        loader: DataLoader yielding (data, target) or (data, indices, target) batches.
        dataset: training dataset, used to build exemplar embeddings when not provided.
        exemplars_embedding: pre-computed prior embeddings; computed once here if None,
            so callers that evaluate multiple splits can pass it in to avoid redundant work.

    Returns:
        Tuple (elbo, re, kl) averaged over the full dataset.
    """
    evaluateed_elbo, evaluate_re, evaluate_kl = 0, 0, 0
    model.eval()

    # Build prior embeddings once — expensive for exemplar_prior (requires a full
    # forward pass over the training set to cache z), so callers can pass it in.
    if exemplars_embedding is None:
        exemplars_embedding = load_all_pseudo_input(args, model, dataset)

    with torch.no_grad():
        for data in loader:
            # Support both (data, target) and (data, indices, target) batch formats.
            if len(data) == 3:
                data, _, _ = data
            else:
                data, _ = data
            data = data.to(args.device)
            x = data
            # x_indices=None disables the exemplar prior's leave-one-out masking.
            # During evaluation we want the full embedding, not a masked version,
            # because there is no risk of the model "cheating" by seeing itself.
            x_indices = None
            x = (x, x_indices)
            loss, RE, KL = model.calculate_loss(x, average=False, exemplars_embedding=exemplars_embedding)
            evaluateed_elbo += loss.sum().item()
            evaluate_re += -RE.sum().item()
            evaluate_kl += KL.sum().item()
    # Divide by total number of data points, not number of batches, for a
    # consistent per-sample average regardless of batch size.
    evaluateed_elbo /= len(loader.dataset)
    evaluate_re /= len(loader.dataset)
    evaluate_kl /= len(loader.dataset)
    return evaluateed_elbo, evaluate_re, evaluate_kl


def visualize_reconstruction(test_samples, model, args, dir):
    n = 25
    with torch.no_grad():
        samples_reconstruction = model.reconstruct_x(test_samples[0:n])

    if args.use_logit:
        test_samples = model.logit_inverse(test_samples)
        samples_reconstruction = model.logit_inverse(samples_reconstruction)
    plot_images(args, test_samples.detach().cpu().numpy()[0:n], dir, 'real', size_x=5, size_y=5)
    # IWAE with resample=True returns K*batch rows — take first n
    plot_images(args, samples_reconstruction.detach().cpu().numpy()[:n], dir, 'reconstructions', size_x=5, size_y=5)


def visualize_generation(dataset, model, args, dir):
    generation_rounds = 1
    for i in range(generation_rounds):
        with torch.no_grad():
            samples_rand = model.generate_x(25, dataset=dataset)
        plot_images(args, samples_rand.detach().cpu().numpy(), dir, 'generations_{}'.format(i), size_x=5, size_y=5)
    if args.prior == 'vampprior':
        pseudo_means = model.means(model.idle_input)
        plot_images(args, pseudo_means[0:25].detach().cpu().numpy(), dir, 'pseudoinputs', size_x=5, size_y=5)


def _visualize_ts_final(test_samples, model, args, dir, train_dataset):
    """Post-training visualization for time series models."""
    n = min(9, len(test_samples))
    pt = getattr(args, 'plot_timesteps', None)

    # 1. Reconstruction overlay
    with torch.no_grad():
        x_recon = model.reconstruct_x(test_samples[:n])
    plot_ts_reconstruction(
        test_samples[:n].cpu().numpy(),
        x_recon.cpu().numpy()[:n],
        dir, 'ts_reconstructions',
        seq_len=args.seq_len, n_samples=n, plot_timesteps=pt,
    )

    # 2. Decoder decomposition (3 samples)
    # Guard: not every model implements decompose_reconstruction (e.g. plain VAE
    # has no additive components), so we check before calling.
    if hasattr(model, 'decompose_reconstruction'):
        n_decomp = min(3, n)
        with torch.no_grad():
            components = model.decompose_reconstruction(test_samples[:n_decomp])
        components_np = {
            k: v.cpu().numpy() if v is not None else None
            for k, v in components.items()
        }
        for i in range(n_decomp):
            plot_ts_decomposition(
                components_np, dir, f'ts_decomposition_{i}',
                seq_len=args.seq_len, sample_idx=i, plot_timesteps=pt,
            )

    # 3. Generation
    with torch.no_grad():
        generated = model.generate_x(N=6, dataset=train_dataset)
    plot_ts_generation(
        generated.cpu().numpy(), dir, 'ts_generations',
        seq_len=args.seq_len, n_samples=6, plot_timesteps=pt,
    )


def load_all_pseudo_input(args, model, dataset):
    """Pre-compute and return prior embeddings used by calculate_loss().

    The returned value ("exemplars_embedding") is passed to calculate_loss() on
    every batch so the prior can be evaluated without re-computing it each time.

    Returns (depending on prior type):
        - exemplar_prior: tuple (z_mean, z_log_var, indices) for the entire training
          set — the model encodes every training point once and caches the result.
          The indices tensor lets calculate_loss() do leave-one-out masking during
          training, but x_indices=None disables that at eval time.
        - vampprior: tuple (z_mean, z_log_var) from encoding the C learnable
          pseudo-inputs through the encoder. Shape: (C, latent_dim).
        - standard: None — N(0,1) prior needs no precomputed parameters.
    """
    if args.prior == 'exemplar_prior':
        # Encode the entire training set once; result is cached on CPU.
        exemplars_z, exemplars_log_var = model.cache_z(dataset)
        # arange gives each exemplar its global index for leave-one-out masking.
        embedding = (exemplars_z, exemplars_log_var, torch.arange(len(exemplars_z)))
    elif args.prior == 'vampprior':
        # Pass learnable pseudo-inputs through the encoder to get the mixture
        # components of the VampPrior.
        pseudo_means = model.means(model.idle_input)
        if 'conv' in args.model_name:
            # Reshape flat pseudo-input vector back to spatial dimensions expected
            # by a convolutional encoder.
            pseudo_means = pseudo_means.view(-1, args.input_size[0], args.input_size[1], args.input_size[2])
        embedding = model.q_z(pseudo_means, prior=True)  # C x M
    elif args.prior == 'standard':
        # Standard Gaussian prior has no parameters to precompute.
        embedding = None
    else:
        raise Exception("wrong name of prior")
    return embedding


def calculate_likelihood(args, model, loader, S=5000, exemplars_embedding=None):
    """Estimate the marginal log-likelihood log p(x) via importance sampling.

    Uses the IWAE / AIS estimator:
        log p(x) ≈ logsumexp_s [ log p(x|z_s) + log p(z_s) - log q(z_s|x) ] - log S
    where z_s ~ q(z|x) for s = 1..S. This is an unbiased estimator of p(x) in
    expectation; larger S gives a tighter (less biased) lower bound.

    Args:
        args: config namespace; uses args.model_name, args.use_logit, args.lambd.
        model: VAE model; calculate_loss() must return per-sample ELBO values when
            average=False (the returned 'prob' is the negative ELBO, i.e. the loss).
        loader: DataLoader for the evaluation set.
        S: number of importance samples per data point. Higher S → tighter bound
            at the cost of O(S) memory and compute.
        exemplars_embedding: pre-computed prior parameters (see load_all_pseudo_input).

    Returns:
        Scalar float: average *negative* log-likelihood over the dataset (bits or nats
        depending on model convention). Lower is better.
    """
    likelihood_test = []
    # Evaluate one data point at a time so we can expand it to S copies without
    # running out of memory. With S=5000 samples, a batch of even 2 images would
    # require 10 000 forward passes simultaneously.
    batch_size_evaluation = 1
    auxilary_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=batch_size_evaluation)
    N = len(auxilary_loader)
    running_nll = 0.0
    with tqdm(total=N, desc=f"LL estimation (S={S})", unit="sample") as pbar:
        for index, (data, _) in enumerate(auxilary_loader):
            data = data.to(args.device)
            x = data.expand(S, data.size(1))  # (S, D)
            with torch.no_grad():
                if args.model_name == 'pixelcnn':
                    BS = S//100
                    prob = []
                    for i in range(BS):
                        bx = x[i*100:(i+1)*100]
                        x_indices = None
                        bprob, _, _ = model.calculate_loss((bx, x_indices), exemplars_embedding=exemplars_embedding)
                        prob.append(bprob)
                    prob = torch.cat(prob, dim=0)
                else:
                    x_indices = None
                    prob, _, _ = model.calculate_loss((x, x_indices), exemplars_embedding=exemplars_embedding)

            likelihood_x = logsumexp(-prob.cpu().numpy())

            if model.args.use_logit:
                lambd = torch.tensor(model.args.lambd).float()
                likelihood_x -= (-F.softplus(-x) - F.softplus(x)\
                                 - torch.log((1 - 2 * lambd)/256)).sum(dim=1).cpu().numpy()

            likelihood_test.append(likelihood_x - np.log(len(prob)))

            running_nll = -np.mean(likelihood_test)
            pbar.update(1)
            if index % 50 == 0:
                pbar.set_postfix(NLL=f"{running_nll:.2f}")

    likelihood_test = np.array(likelihood_test)
    # Return the mean *negative* log-likelihood (lower is better; convention
    # matches how ELBO is reported as a positive loss in this codebase).
    return -np.mean(likelihood_test)


_LL_METRIC_COLUMNS = [
    "test_nll", "test_elbo", "train_elbo", "val_elbo",
    "test_re", "test_kl", "S", "epochs_trained",
]


def save_ll_metrics_csv(metrics: dict, output_path) -> None:
    """Save LL estimation metrics to CSV."""
    df = pd.DataFrame([metrics], columns=_LL_METRIC_COLUMNS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.4f")


def final_evaluation(args, model, train_loader, val_loader, test_loader, dir, compute_ll=True):
    """Comprehensive post-training evaluation: ELBO, likelihood, visualization.

    Assumes the model is already loaded with best checkpoint weights.
    Caller is responsible for calling load_model() before this function.

    Flow:
        1. Visualize reconstructions and generations (or TS plots).
        2. Compute ELBO on train / val / test splits.
        3. Compute importance-sampled log-likelihood on the test set (expensive).
        4. Log and save all metrics.
        5. Optionally run clustering analysis on test-set latent codes.

    Args:
        args: config namespace; uses args.S, args.cluster, args.input_size, etc.
        model: trained VAE (already in eval mode after load_model).
        train_loader / val_loader / test_loader: DataLoaders for the three splits.
        dir: output directory (str) for saving plots, logs, and metric tensors.

    Returns:
        Tuple (test_elbo, test_re, test_kl).
    """
    model.eval()
    # Pre-compute prior embeddings once here and reuse across all evaluate_loss()
    # calls; avoids re-encoding the training set for every split evaluation.
    exemplars_embedding = load_all_pseudo_input(args, model, train_loader.dataset)

    # ----- visualization -----
    # Dispatch on input dimensionality: 3-element input_size → image (C, H, W);
    # 2-element → time series (seq_len, n_features).
    if len(args.input_size) == 3:
        test_samples = next(iter(test_loader))[0].to(args.device)
        visualize_reconstruction(test_samples, model, args, dir)
        visualize_generation(train_loader.dataset, model, args, dir)
    elif len(args.input_size) == 2:
        test_samples = next(iter(test_loader))[0].to(args.device)
        _visualize_ts_final(test_samples, model, args, dir, train_loader.dataset)

    # ----- ELBO on all splits -----
    test_elbo, test_re, test_kl = evaluate_loss(
        args, model, test_loader,
        dataset=train_loader.dataset, exemplars_embedding=exemplars_embedding)
    val_elbo, _, _ = evaluate_loss(
        args, model, val_loader,
        dataset=train_loader.dataset, exemplars_embedding=exemplars_embedding)
    train_elbo, _, _ = evaluate_loss(
        args, model, train_loader,
        dataset=train_loader.dataset, exemplars_embedding=exemplars_embedding)

    # ----- importance-sampled log-likelihood (expensive, optional) -----
    test_log_likelihood = None
    if compute_ll:
        print('Computing test log-likelihood (S={})...'.format(args.S))
        test_log_likelihood = calculate_likelihood(
            args, model, test_loader,
            exemplars_embedding=exemplars_embedding, S=args.S)

    # ----- report -----
    if test_log_likelihood is not None:
        final_txt = (
            'FINAL EVALUATION ON TEST SET\n'
            'LogL (TEST):  {:.2f}\n'
            'ELBO (TEST):  {:.2f}\n'
            'ELBO (TRAIN): {:.2f}\n'
            'ELBO (VALID): {:.2f}\n'
            'RE:           {:.2f}\n'
            'KL:           {:.2f}'
        ).format(test_log_likelihood, test_elbo, train_elbo, val_elbo, test_re, test_kl)
    else:
        final_txt = (
            'FINAL EVALUATION ON TEST SET (LL skipped)\n'
            'ELBO (TEST):  {:.2f}\n'
            'ELBO (TRAIN): {:.2f}\n'
            'ELBO (VALID): {:.2f}\n'
            'RE:           {:.2f}\n'
            'KL:           {:.2f}'
        ).format(test_elbo, train_elbo, val_elbo, test_re, test_kl)

    print(final_txt)

    # save metrics (torch format for backward compatibility)
    if test_log_likelihood is not None:
        torch.save(test_log_likelihood, dir + args.model_name + '.test_log_likelihood')
    torch.save(test_elbo, dir + args.model_name + '.test_loss')
    torch.save(test_re, dir + args.model_name + '.test_re')
    torch.save(test_kl, dir + args.model_name + '.test_kl')

    # save CSV metrics
    epochs_trained = getattr(args, 'epochs_trained', None)
    ll_metrics = {
        "test_nll": test_log_likelihood,
        "test_elbo": test_elbo,
        "train_elbo": train_elbo,
        "val_elbo": val_elbo,
        "test_re": test_re,
        "test_kl": test_kl,
        "S": args.S if compute_ll else None,
        "epochs_trained": epochs_trained,
    }
    save_ll_metrics_csv(ll_metrics, dir + 'll_metrics.csv')
    print(f'Results saved to {dir}ll_metrics.csv')

    # ----- clustering analysis (optional — enabled with --cluster flag) -----
    if getattr(args, 'cluster', False):
        from utils.clustering import cluster_latent
        all_data, all_labels = [], []
        for batch in test_loader:
            all_data.append(batch[0])
            all_labels.append(batch[-1])
        all_data = torch.cat(all_data).to(args.device)
        all_labels = torch.cat(all_labels)
        cluster_latent(args, model, all_data, all_labels, output_dir=dir)

    # ----- reconstruction visualization (optional — enabled with --recon_viz flag) -----
    if getattr(args, 'recon_viz', False):
        from utils.visual_recon import plot_reconstruction_likelihood
        plot_reconstruction_likelihood(args, model, test_loader, dir)

    # ----- reference-based generation (optional — enabled with --generate flag) -----
    if getattr(args, 'generate', False):
        from utils.plot_images import generate_fancy_grid
        with torch.no_grad():
            exemplars_n = 50
            selected_indices = torch.randint(
                low=0, high=args.training_set_size, size=(exemplars_n,))
            reference_images, indices, labels = train_loader.dataset[selected_indices]
            per_exemplar = 11
            generated = model.reference_based_generation_x(
                N=per_exemplar, reference_image=reference_images)
            generated = generated.reshape(-1, per_exemplar, *args.input_size)
            if getattr(args, 'use_logit', False):
                reference_images = model.logit_inverse(reference_images)
            generate_fancy_grid(args, dir, reference_images, generated)
        print(f"Generation complete -> {dir}generated/")

    return test_elbo, test_re, test_kl


def compute_mean_variance_per_dimension(model, data_loader, device, threshold=0.01):
    """Count active latent dimensions via Var_x[E[z|x]] > threshold.

    A dimension is "active" if the encoder uses it to encode input-dependent
    information rather than collapsing to the prior.

    Args:
        model:       VAE model with a q_z(x) method returning (mean, logvar).
        data_loader: DataLoader yielding (data, ...) batches.
        device:      torch.device for input tensors.
        threshold:   Variance threshold for "active" (default 0.01).

    Returns:
        Tuple (active_count, variances) where variances is a 1D numpy array
        of shape (z_dim,) with per-dimension Var_x[E[z|x]] values.
    """
    means = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            data = batch[0].to(device)
            mean, _ = model.q_z(data)
            means.append(mean)
    means = torch.cat(means, dim=0).cpu().numpy()
    variances = np.var(means, axis=0)
    active_count = int(np.sum(variances > threshold))
    return active_count, variances


# ======================================================================================================================
# Evaluation for the run.py path (data, target) format
# ======================================================================================================================
def evaluate_vae(args, model, train_loader, data_loader, epoch, dir, mode,
                 _fixed_samples={}):
    """Evaluate model on one split for a single epoch; save reconstruction images.

    Called once per epoch on the validation set during training to track progress.
    Also used at test time (mode='test') in the run.py path.

    Args:
        args: config namespace.
        model: VAE model.
        train_loader: loader for the training set — used to build exemplar embeddings.
        data_loader: loader for the split being evaluated (val or test).
        epoch: current epoch number (used to name saved reconstruction images).
        dir: base output directory.
        mode: 'validation' or 'test'; controls whether fixed samples are captured/plotted.
        _fixed_samples: mutable default dict — a well-known Python trick to persist
            state across calls without a class. The dict is shared across all calls
            in the same Python session, so the fixed reference batch is captured on
            the first validation epoch and reused every subsequent epoch for a
            consistent side-by-side comparison of reconstructions over time.

    Returns:
        Tuple (elbo, re, kl) averaged over data_loader.dataset.
    """
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0

    # put the model in evaluation mode – no gradients needed
    model.eval()

    # ----- capture fixed samples once for consistent reconstruction plots -----
    # We use len(args.input_size) to distinguish image (3D) from time series (2D).
    _is_image = len(args.input_size) == 3
    _is_timeseries = len(args.input_size) == 2
    if _is_image and mode == 'validation' and 'x' not in _fixed_samples:
        # Grab the very first batch of validation data; keep only 9 samples so the
        # plot grid is 3×3. These same 9 samples will be reconstructed every epoch.
        sample_batch = next(iter(data_loader))
        _fixed_samples['x'] = sample_batch[0][:9].clone()
        os.makedirs(dir + 'reconstruction/', exist_ok=True)
        plot_images(
            args,
            _fixed_samples['x'].cpu().numpy(),
            dir + 'reconstruction/',
            'real',
            size_x=3, size_y=3
        )
    if _is_timeseries and mode == 'validation' and 'x' not in _fixed_samples:
        sample_batch = next(iter(data_loader))
        # Clamp to actual batch size (small datasets may have fewer than 6 val rows)
        n_ts_samples = min(6, len(sample_batch[0]))
        _fixed_samples['x'] = sample_batch[0][:n_ts_samples].clone()
        os.makedirs(dir + 'reconstruction/', exist_ok=True)
        plot_ts_generation(
            _fixed_samples['x'].cpu().numpy(),
            dir + 'reconstruction/', 'real',
            seq_len=args.seq_len, n_samples=n_ts_samples,
            plot_timesteps=getattr(args, 'plot_timesteps', None),
        )

    # ----- pre-compute exemplar embeddings for exemplar prior -----
    # For vampprior/standard this is cheap (or None); for exemplar_prior it encodes
    # the full training set, so we do it once outside the batch loop below.
    if args.prior == 'exemplar_prior':
        with torch.no_grad():
            exemplars_embedding = load_all_pseudo_input(args, model, train_loader.dataset)
    else:
        exemplars_embedding = None

    # ----- evaluation loop over the data loader -----
    for batch_idx, (data, target) in enumerate(data_loader):
        # move tensors to GPU if requested
        if args.cuda:
            data, target = data.to(args.device), target.to(args.device)

        with torch.no_grad():
            x = data
            # calculate_loss() always expects a (x, x_indices) tuple.
            # x_indices drives the exemplar prior's leave-one-out masking: during
            # training each sample's own index is excluded from the prior mixture.
            # At eval time we disable this by passing dummy zero indices —
            # leave-one-out is not needed because we are not optimising the model.
            dummy_indices = torch.zeros(data.size(0), 1, dtype=torch.long, device=data.device)
            # calculate loss function — per-sample losses, summed below
            loss, RE, KL = model.calculate_loss(
                (x, dummy_indices), average=False,
                exemplars_embedding=exemplars_embedding)

        # accumulate per-sample metrics
        evaluate_loss += loss.sum().item()
        evaluate_re += -RE.sum().item()
        evaluate_kl += KL.sum().item()

    # ----- save reconstruction of fixed samples every epoch -----
    if _is_image and mode == 'validation' and 'x' in _fixed_samples:
        fixed_x = _fixed_samples['x'].to(args.device)
        with torch.no_grad():
            x_mean = model.reconstruct_x(fixed_x)
        # IWAE with resample=True returns K*batch rows — take first N
        n = len(_fixed_samples['x'])
        epoch_str = f"{epoch:04d}"
        plot_images(
            args,
            x_mean.data.cpu().numpy()[:n],
            dir + 'reconstruction/',
            epoch_str,
            size_x=3, size_y=3
        )
    if _is_timeseries and mode == 'validation' and 'x' in _fixed_samples:
        fixed_x = _fixed_samples['x'].to(args.device)
        with torch.no_grad():
            x_mean = model.reconstruct_x(fixed_x)
        n = len(_fixed_samples['x'])
        epoch_str = f"{epoch:04d}"
        plot_ts_reconstruction(
            _fixed_samples['x'].cpu().numpy()[:n],
            x_mean.data.cpu().numpy()[:n],
            dir + 'reconstruction/', epoch_str,
            seq_len=args.seq_len, n_samples=n,
            plot_timesteps=getattr(args, 'plot_timesteps', None),
        )

    # ----- final metric aggregation (divide by total samples, not batches) -----
    N = len(data_loader.dataset)
    evaluate_loss /= N
    evaluate_re /= N
    evaluate_kl /= N

    return evaluate_loss, evaluate_re, evaluate_kl
