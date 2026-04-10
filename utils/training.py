from __future__ import print_function
import torch

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# ======================================================================================================================
# Warm-up beta schedule
# ======================================================================================================================
def set_beta(args, epoch):
    """Linearly increase beta from 0 to 1 over the first ``args.warmup`` epochs.

    Beta scales the KL term in the ELBO: loss = RE + beta * KL.
    Starting at beta=0 lets the encoder warm up with pure reconstruction
    pressure before the prior regularisation kicks in, which prevents
    posterior collapse in the early training phase.

    Args:
        args:  Argument namespace; reads ``args.warmup`` (int, 0 = no warmup).
        epoch: Current training epoch (1-indexed).

    Returns:
        beta (float): KL weight in [0.0, 1.0].
    """
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1. * epoch / args.warmup
        if beta > 1.:
            beta = 1.
    return beta


# ======================================================================================================================
# Training loop for density_estimation.py path (data, indices, target) format
# ======================================================================================================================
def train_one_epoch(epoch, args, train_loader, model, optimizer):
    """Run one full epoch of training for the density_estimation.py entry point.

    Data loader must yield ``(data, indices, target)`` triples, where
    ``indices`` are the per-sample positions in the full training set — required
    by the exemplar prior for leave-one-out (LOO) masking.

    Args:
        epoch:        Current epoch number (1-indexed); used for beta warmup.
        args:         Argument namespace.  Key fields: ``dynamic_binarization``,
                      ``approximate_prior``, ``device``, ``warmup``.
        train_loader: DataLoader yielding ``(data, indices, target)`` batches.
        model:        VAE model with a ``calculate_loss`` and ``cache_z`` method.
        optimizer:    PyTorch optimiser.

    Returns:
        Tuple ``(train_loss, train_re, train_kl)`` — per-batch averages over the epoch.
        Note: ``train_re`` is stored as the *negative* RE so that all three values
        are non-negative and directly comparable in logs.
    """
    # set loss to 0
    train_loss, train_re, train_kl = 0, 0, 0
    # set model in training mode
    model.train()
    # warm-up beta
    beta = set_beta(args, epoch)
    print('beta: {}'.format(beta))

    # ----- approximate-prior cache -----
    # When approximate_prior is True, the exemplar prior uses nearest-neighbour
    # search in latent space instead of random exemplar sampling.  Running the
    # entire dataset through the encoder once per epoch (cache_z) is much cheaper
    # than re-encoding a fresh random subset on every batch.
    # torch.no_grad() avoids building a computation graph for all N training
    # examples — we only need the values, not gradients through the cache pass.
    if args.approximate_prior is True:
        with torch.no_grad():
            cached_z, cached_log_var = model.cache_z(train_loader.dataset)
            # cached_z:       (N, z_dim) — encoder means for every training sample
            # cached_log_var: (N, z_dim) — encoder log-variances for every training sample
            cache = (cached_z, cached_log_var)
    else:
        cache = None

    # ----- main training loop -----
    # start training
    for batch_idx, (data, indices, target) in enumerate(train_loader):
        # data:    (batch, C, H, W) or (batch, D) — raw inputs
        # indices: (batch,) — positions in the full training set; needed for LOO masking
        # target:  (batch,) — class labels (unused in VAE loss, kept for compatibility)
        data, indices, target = data.to(args.device), indices.to(args.device), target.to(args.device)

        # dynamic binarization
        if args.dynamic_binarization:
            # Re-sample a fresh binary mask each time the same image appears.
            # Treating pixel intensities as Bernoulli probabilities turns the
            # greyscale dataset into a stochastic binary one, which prevents the
            # decoder from memorising exact grey values.
            x = torch.bernoulli(data)
        else:
            x = data

        # pack data with indices for exemplar prior leave-one-out masking
        # calculate_loss unpacks this tuple; indices tell the exemplar prior
        # which training-set entries are the current batch so they can be
        # excluded from the candidate exemplar pool (leave-one-out).
        x = (x, indices)
        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        # average=True: loss is divided by batch size inside calculate_loss
        # cache: pre-encoded dataset embeddings (None when approximate_prior=False)
        loss, RE, KL = model.calculate_loss(x, beta, average=True, cache=cache, dataset=train_loader.dataset)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        # ----- metric accumulation -----
        # accumulate metrics (no gradient needed)
        with torch.no_grad():
            train_loss += loss.data.item()
            train_re += -RE.data.item()   # RE is negative (log-likelihood); negate for readability
            train_kl += KL.data.item()
            # detach cache to prevent memory leak across batches
            # loss.backward() accumulates gradients for every tensor in the
            # computation graph.  If cache still holds references created before
            # the backward pass, subsequent detach-free calls would keep extending
            # the graph across batches, leaking memory.  Detaching cuts those ties
            # while keeping the numerical values intact for the next batch.
            if cache is not None:
                cache = (cache[0].detach(), cache[1].detach())

    # calculate final loss
    train_loss /= len(train_loader)
    train_re /= len(train_loader)
    train_kl /= len(train_loader)
    return train_loss, train_re, train_kl


# ======================================================================================================================
# Training loop for run.py path (data, target) format
# ======================================================================================================================
def train_vae(epoch, args, train_loader, model, optimizer, beta=None):
    """Run one full epoch of training for the run.py / perform_experiment.py entry point.

    Mirrors ``train_one_epoch`` but differs in two ways:
      - ``beta`` may be injected by an external BetaScheduler (perform_experiment.py path)
        rather than always being computed from ``args.warmup``.
      - The ``args.cuda`` flag guards the ``.to(device)`` call (older-style check
        kept for backwards compatibility with the run.py arg namespace).

    Data loader must yield ``(data, indices, target)`` triples.

    Args:
        epoch:        Current epoch number (1-indexed).
        args:         Argument namespace.  Key fields: ``cuda``, ``device``,
                      ``dynamic_binarization``, ``approximate_prior``, ``warmup``.
        train_loader: DataLoader yielding ``(data, indices, target)`` batches.
        model:        VAE model with ``calculate_loss`` and ``cache_z`` methods.
        optimizer:    PyTorch optimiser.
        beta (float, optional): KL weight override.  When None, ``set_beta`` is
                      called to compute it from epoch and ``args.warmup``.

    Returns:
        Tuple ``(model, train_loss, train_re, train_kl)`` — the updated model and
        per-batch averages over the epoch.  ``train_re`` is negated (see below).
    """
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    # beta can be passed from BetaScheduler (perform_experiment.py path)
    # or computed here as fallback (density_estimation.py path)
    if beta is None:
        beta = set_beta(args, epoch)

    # ----- approximate-prior cache -----
    # Pre-encode the full training set once per epoch so that the approximate
    # nearest-neighbour exemplar search (get_approximate_nearest_exemplars) has
    # cached embeddings to query against.  Without this, every batch would have
    # to re-encode the entire dataset, making training O(N) times more expensive.
    # getattr with a default handles the case where approximate_prior was not added
    # to args (e.g. older checkpoints loaded before the flag was introduced).
    if getattr(args, 'approximate_prior', False):
        with torch.no_grad():
            cached_z, cached_log_var = model.cache_z(train_loader.dataset)
            # cached_z:       (N, z_dim) — encoder means for the full training set
            # cached_log_var: (N, z_dim) — encoder log-variances for the full training set
            cache = (cached_z, cached_log_var)
    else:
        cache = None

    # ----- main training loop -----
    # start training
    # train loader yields (data, indices, target)
    for batch_idx, (data, indices, target) in enumerate(train_loader):
        # data:    (batch, C, H, W) or (batch, D)
        # indices: (batch,) — training-set positions, forwarded to calculate_loss
        #          for exemplar LOO masking; also used to update the approximate cache
        # target:  (batch,) — class labels (not used in the VAE loss itself)
        if args.cuda:
            data, indices, target = data.to(args.device), indices.to(args.device), target.to(args.device)

        # dynamic binarization
        if args.dynamic_binarization:
            # Fresh Bernoulli sample per epoch so the model sees stochastic binary
            # inputs rather than fixed grey values — acts as data augmentation and
            # forces the model to learn uncertainty over pixel states.
            x = torch.bernoulli(data)
        else:
            x = data

        # pack data with indices (needed by calculate_loss for all prior types)
        # Even for standard/VampPrior, calculate_loss accepts this tuple format;
        # indices are only consumed when prior == 'exemplar_prior'.
        x = (x, indices)
        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        # average=True: ELBO components are divided by batch size inside calculate_loss
        # cache: pre-computed dataset embeddings (None when approximate_prior=False)
        # dataset: full training TensorDataset, used to draw random exemplar samples
        #          when approximate_prior=False (random strategy in get_exemplar_set)
        loss, RE, KL = model.calculate_loss(x, beta, average=True,
                                            cache=cache, dataset=train_loader.dataset)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        # ----- metric accumulation -----
        # accumulate metrics (no gradient needed)
        with torch.no_grad():
            train_loss += loss.item()
            train_re += -RE.item()   # RE is negative log-likelihood; negate so the logged value is positive
            train_kl += KL.item()
            # detach cache to prevent memory leak across batches
            # After loss.backward() the autograd graph anchored to cache tensors
            # is no longer needed.  Calling .detach() breaks the link between the
            # cache and any previously-computed graphs, preventing GPU memory from
            # growing monotonically over the epoch.  The tensor values are preserved
            # so the next batch can still use them for nearest-neighbour lookup.
            if cache is not None:
                cache = (cache[0].detach(), cache[1].detach())

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl
