import json
import logging
import os
import torch
import math
import time

from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.model_io import save_model, load_model

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
# Beta (KL weight) scheduler — follows PyTorch scheduler API conventions
# ======================================================================================================================
class BetaScheduler:
    """KL weight warm-up following PyTorch scheduler API conventions.

    Linearly ramps beta from 0 to 1 over ``warmup_epochs``, then holds at 1.
    Prevents posterior collapse early in training by starting with only
    reconstruction loss.
    """
    def __init__(self, warmup_epochs):
        self.warmup_epochs = warmup_epochs
        self._epoch = 0

    def step(self):
        self._epoch += 1

    def get_beta(self):
        if self.warmup_epochs == 0:
            return 1.0
        return min(1.0, self._epoch / self.warmup_epochs)

    def state_dict(self):
        return {'epoch': self._epoch, 'warmup_epochs': self.warmup_epochs}

    def load_state_dict(self, state):
        self._epoch = state['epoch']
        self.warmup_epochs = state['warmup_epochs']


# ======================================================================================================================

def _prompt_auto_z_size(active, total):
    """Display AU results and ask the user whether to restart training.

    Returns True if the user accepts, False otherwise.
    """
    w = 56  # inner width between the | borders
    lines = [
        'Latent space analysis complete',
        f'Active units: {active} / {total}',
        f'Recommended z1_size: {active}',
        '',
        f'Restart training with z1_size={active}? [y/n]',
    ]
    border = '+' + '-' * (w + 2) + '+'
    print('\n' + border)
    for line in lines:
        print(f'|  {line:<{w}}|')
    print(border)
    while True:
        response = input('> ').strip().lower()
        if response in ('y', 'n'):
            return response == 'y'
        print("Please enter 'y' or 'n'.")


def experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name='vae'):
    """Run the full training + validation loop, then call final evaluation.

    Manages three interacting mechanisms — beta warm-up, ReduceLROnPlateau
    scheduling, and early stopping — before loading the best checkpoint and
    delegating to ``final_evaluation``.

    Args:
        args:          Parsed CLI / config namespace (see args_reference.md).
        train_loader:  DataLoader for training split; also passed to evaluate()
                       because the exemplar prior needs access to the full
                       training set for leave-one-out masking.
        val_loader:    DataLoader for validation split.
        test_loader:   DataLoader for test split (used only in final_evaluation).
        model:         Instantiated VAE model (subclass of BaseModel).
        optimizer:     Torch optimizer (Adam, AdamNormGrad, …).
        dir:           Output directory (trailing slash expected); checkpoints,
                       configs, and loss histories are written here.
        model_name:    Informational; not used inside the function body — kept
                       for API compatibility with callers.

    Returns:
        None.  Side effects: writes checkpoints, .config, and loss history
        tensors to ``dir``; calls final_evaluation which writes test metrics
        and reconstruction images.
    """
    # Deferred import keeps top-level imports lean and lets callers swap
    # training/evaluation implementations without touching this module.
    from utils.training import train_vae as train
    from utils.evaluation import evaluate_vae as evaluate

    log = logging.getLogger(__name__)

    # --- AU monitor setup ---
    if getattr(args, 'auto_z_size', False):
        from utils.active_units import ActiveUnitsMonitor
        from utils.evaluation import compute_mean_variance_per_dimension
        au_monitor = ActiveUnitsMonitor(
            check_interval=args.au_check_interval,
            stability_count=args.au_stability_count,
        )
    else:
        au_monitor = None

    # ==================================================================
    # Training loop (with optional AU-driven restart)
    # ==================================================================
    # When --auto_z_size is active the outer while loop allows a single
    # restart: after AU stabilizes the user is prompted to accept a new
    # z1_size; on acceptance a fresh model + optimizer are created and
    # training re-enters the while loop. Without --auto_z_size (or
    # after the restart) restart_training stays False and the while
    # loop executes exactly once.
    # ==================================================================

    # ------------------------------------------------------------------
    # Load-only mode: skip training, go straight to final_evaluation.
    # Triggered when a pretrained model was loaded via --dir (args.dir
    # is set by run.py before calling experiment_vae).
    # ------------------------------------------------------------------
    if getattr(args, 'dir', None) is not None:
        train_loss_path = dir + args.model_name + '.train_loss'
        try:
            train_loss_history = torch.load(train_loss_path, weights_only=True)
        except (FileNotFoundError, Exception):
            train_loss_history = []
        args.epochs_trained = len(train_loss_history)

        # Resolve --ll flag (same logic as the training path below)
        if args.ll is None:
            compute_ll = True   # load mode default → compute
        else:
            compute_ll = args.ll

        from utils.evaluation import final_evaluation
        final_evaluation(
            args, model, train_loader, val_loader, test_loader, dir,
            compute_ll=compute_ll)
        return

    restart_training = True
    while restart_training:
        restart_training = False

        # ==============================================================
        # Scheduling configuration
        # ==============================================================
        # To customize: edit parameters below. No CLI args needed —
        # these are stable defaults that rarely change between
        # experiments.
        # ==============================================================

        # --- Beta (KL weight) warm-up ---
        # Ramps KL weight linearly 0 → 1 over warmup epochs, preventing
        # posterior collapse early in training. During warmup, early
        # stopping is also suppressed (see loop below).
        beta_scheduler = BetaScheduler(warmup_epochs=args.warmup)

        # --- LR scheduling ---
        # ReduceLROnPlateau monitors val_loss (already computed each
        # epoch) and halves LR when validation loss stops improving.
        #
        # Interaction with early stopping:
        #   scheduler patience (20) < early_stopping_epochs (default 50),
        #   so a plateau first triggers LR reduction. If that doesn't
        #   help, early stopping eventually fires.
        #
        # --- Alternative LR schedulers (uncomment to replace) ---------
        #
        # Cosine decay (needs T_max; less useful with early stopping):
        #   from torch.optim.lr_scheduler import CosineAnnealingLR
        #   lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        #   # change lr_scheduler.step(val_loss_epoch) -> lr_scheduler.step()
        #
        # One-cycle policy (aggressive; for short fixed-length runs):
        #   from torch.optim.lr_scheduler import OneCycleLR
        #   lr_scheduler = OneCycleLR(optimizer, max_lr=args.lr * 10,
        #                             epochs=args.epochs,
        #                             steps_per_epoch=len(train_loader))
        #   # call lr_scheduler.step() after each *batch*, not epoch
        #
        # No scheduling:
        #   lr_scheduler = None
        #   # guard the step: if lr_scheduler: lr_scheduler.step(...)
        # ==============================================================
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,             # halve LR on reduction
            patience=20,            # epochs before reducing LR
            min_lr=1e-6,            # LR floor
            cooldown=5,             # epochs after reduction before counting again
        )

        # checkpoint paths
        # checkpoint_temp is written first; only renamed to
        # checkpoint_best on success, so a crash mid-save never
        # corrupts the best checkpoint.
        checkpoint_path_save = dir + 'checkpoint_temp.pth'
        checkpoint_path_best = dir + 'checkpoint_best.pth'

        # Persist the full args namespace so the model can be reloaded
        # later without re-specifying CLI flags (used by load_model.py
        # and analysis.py).
        torch.save(args, dir + args.model_name + '.config')

        # Intentionally a large finite number rather than float('inf')
        # so that the first epoch always registers as an improvement
        # and triggers a save.
        best_loss = 100000.
        e = 0
        train_loss_history = []
        train_re_history = []
        train_kl_history = []

        val_loss_history = []
        val_re_history = []
        val_kl_history = []

        time_history = []

        # ==============================================================
        # Training schedule overview
        # ==============================================================
        # Three mechanisms govern training dynamics:
        #
        # 1. Beta warm-up (args.warmup epochs, default 100):
        #    KL weight ramps linearly 0 → 1. During warmup, early
        #    stopping is suppressed so the model learns basic
        #    reconstructions first.
        #
        # 2. LR scheduling (ReduceLROnPlateau, configured above):
        #    Monitors val_loss and reduces LR after patience epochs of
        #    no improvement. Active from epoch 1 — no special warmup
        #    handling needed because patience naturally covers early
        #    epochs.
        #
        # 3. Early stopping (args.early_stopping_epochs, default 50):
        #    Stops training after N epochs with no val_loss improvement.
        #    Suppressed during beta warmup (epoch < args.warmup).
        #    Scheduler patience (20) < early_stopping (50) → plateau
        #    first triggers LR reduction, then early stopping.
        # ==============================================================

        for epoch in range(1, args.epochs + 1):

            time_start = time.time()
            beta_scheduler.step()
            beta = beta_scheduler.get_beta()
            model, train_loss_epoch, train_re_epoch, train_kl_epoch = train(epoch, args, train_loader, model,
                                                                                optimizer, beta=beta)
            # train_loss_epoch, train_re_epoch, train_kl_epoch: scalars (mean over batches)

            # train_loader is passed here (not just val_loader) because evaluate()
            # uses it to build the exemplar prior's support set during validation.
            val_loss_epoch, val_re_epoch, val_kl_epoch = evaluate(args, model, train_loader, val_loader, epoch, dir, mode='validation')
            # val_loss_epoch, val_re_epoch, val_kl_epoch: scalars (mean over val set)
            time_end = time.time()

            time_elapsed = time_end - time_start

            # appending history
            train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
                train_kl_epoch)
            val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
                val_kl_epoch)
            time_history.append(time_elapsed)

            # printing results
            print('\nEpoch: {}/{}, Time elapsed: {:.2f}s\n'
                '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
                'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
                '--> Early stopping: {}/{} (BEST: {:.2f})\n'
                '    Beta: {:.2f}   LR: {:.4e}'.format(
                epoch, args.epochs, time_elapsed,
                train_loss_epoch, train_re_epoch, train_kl_epoch,
                val_loss_epoch, val_re_epoch, val_kl_epoch,
                e, args.early_stopping_epochs, best_loss,
                beta, optimizer.param_groups[0]['lr']
            ))

            # step LR scheduler (requires val_loss metric)
            lr_scheduler.step(val_loss_epoch)

            # ================ AU monitoring ================
            if au_monitor is not None and au_monitor.should_check(epoch, args.warmup):
                active, _ = compute_mean_variance_per_dimension(
                    model, val_loader, args.device, args.au_threshold)
                au_monitor.record(active)
                print(f'[AU] Active units: {active}/{args.z1_size}')

                if au_monitor.is_stable():
                    if active >= args.z1_size:
                        print(f'[AU] All {args.z1_size} dimensions are active. '
                              'Consider training with a larger z1_size.')
                        result = au_monitor.get_result(args.z1_size, epoch, user_accepted=None)
                        with open(dir + 'auto_z_size_result.json', 'w') as f:
                            json.dump(result, f, indent=2)
                        au_monitor.disable()
                    else:
                        if active < 5:
                            print(f'[AU] Warning: Only {active} dims active '
                                  '— possible posterior collapse or insufficient warmup.')

                        accepted = _prompt_auto_z_size(active, args.z1_size)
                        result = au_monitor.get_result(args.z1_size, epoch, accepted)

                        with open(dir + 'auto_z_size_result.json', 'w') as f:
                            json.dump(result, f, indent=2)

                        if accepted:
                            original_z1 = args.z1_size
                            args.z1_size = active
                            args.auto_z_size = False
                            log.info('Restarting training with z1_size=%d (was %d)',
                                     active, original_z1)

                            from utils.model_io import importing_model
                            from utils.optimizer import AdamNormGrad
                            VAE = importing_model(args)
                            model = VAE(args)
                            model.to(args.device)
                            optimizer = AdamNormGrad(model.parameters(), lr=args.lr)
                            au_monitor = None
                            restart_training = True
                            break
                        else:
                            au_monitor.disable()
            # ================ END: AU monitoring ================

            # ----- early stopping + checkpoint -----

            if val_loss_epoch < best_loss:
                e = 0
                best_loss = val_loss_epoch
                # Checkpoint stores state_dicts rather than the whole model object
                # so it is optimizer- and scheduler-aware: resuming training after
                # an interrupt restores LR, momentum buffers, beta, and the best
                # loss seen so far.  Loss histories are NOT checkpointed — they are
                # reconstructed from the .train_loss/.val_loss files saved at the end.
                content = {
                    'epoch':          epoch,
                    'state_dict':     model.state_dict(),
                    'optimizer':      optimizer.state_dict(),
                    'lr_scheduler':   lr_scheduler.state_dict(),
                    'beta_scheduler': beta_scheduler.state_dict(),
                    'best_loss':      best_loss,
                    'e':              e,
                }
                save_model(checkpoint_path_save, checkpoint_path_best, content)
                print('Model saved...')
            else:
                e += 1
                # Suppress early stopping during beta warm-up: the model has not
                # yet seen the full KL penalty, so val_loss is not yet a stable
                # signal.  Resetting e here means the patience counter only starts
                # accumulating once beta == 1 (i.e., epoch >= args.warmup).
                if epoch < args.warmup:
                    e = 0
                # Strict '>' means early stopping fires after (early_stopping_epochs + 1)
                # non-improving epochs, giving one extra epoch of grace.
                if e > args.early_stopping_epochs:
                    break

            # NaN guard: a NaN loss means gradients have exploded; continuing
            # would corrupt the best checkpoint loaded afterwards, so abort early.
            # Placed after the early-stopping block so the epoch counter e is still
            # updated correctly for any logging that follows.
            if math.isnan(val_loss_epoch):
                break

        if restart_training:
            continue

        # ---- AU: log if never stabilized ----
        if au_monitor is not None and not au_monitor.is_stable() and len(au_monitor.au_history) > 0:
            print(f'[AU] Did not stabilize within training. '
                  f'Last reading: {au_monitor.latest} active units.')

    # ======================================================================
    # Final evaluation
    # ======================================================================

    # Restore the best weights found during training before evaluating on
    # the test set.  If training diverged (NaN) and no checkpoint was ever
    # written, load_model will raise — intentional, as results would be
    # meaningless.
    load_model(checkpoint_path_best, model, optimizer)

    log.info("----------------------------------------")
    log.info("Arguments:\n%s", json.dumps(vars(args), default=str))

    # Deferred import mirrors the pattern used for train/evaluate above and
    # avoids a circular dependency at module load time.
    # final_evaluation runs ELBO on all three splits, optionally estimates
    # likelihood (expensive), saves reconstruction images, and writes test
    # metrics to disk — so test_loss/re/kl are already persisted after this call.
    from utils.evaluation import final_evaluation

    # Record how many epochs were actually trained (for CSV output)
    args.epochs_trained = len(train_loss_history)

    # Resolve --ll flag: None means skip LL after training (expensive; use --ll to force)
    if args.ll is None:
        compute_ll = False   # train mode default → skip
    else:
        compute_ll = args.ll

    test_loss, test_re, test_kl = final_evaluation(
        args, model, train_loader, val_loader, test_loader, dir,
        compute_ll=compute_ll)

    # Save per-epoch training curves so analysis.py / external notebooks can
    # plot them.  Test metrics are intentionally excluded here because
    # final_evaluation already wrote them to dedicated files.
    torch.save(train_loss_history, dir + args.model_name + '.train_loss')
    torch.save(train_re_history, dir + args.model_name + '.train_re')
    torch.save(train_kl_history, dir + args.model_name + '.train_kl')
    torch.save(val_loss_history, dir + args.model_name + '.val_loss')
    torch.save(val_re_history, dir + args.model_name + '.val_re')
    torch.save(val_kl_history, dir + args.model_name + '.val_kl')

    # Machine-readable output for external tools (e.g. calcium_analysis pipeline).
    # Absolute path so callers don't need to know the training cwd.
    print(f"MODEL_DIR: {os.path.abspath(dir.rstrip('/'))}")
