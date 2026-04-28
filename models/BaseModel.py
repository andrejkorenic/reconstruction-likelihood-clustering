from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from utils.nn import normal_init, NonLinear
from utils.distributions import log_normal_diag_vectorized
import math
from utils.nn import he_init
from utils.distributions import pairwise_distance
from utils.distributions import log_bernoulli, log_normal_diag, log_normal_standard, log_logistic_256
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self, args):
        """Initialize shared VAE components: prior parameters, decoder output heads, and model layers.

        Args:
            args: Namespace with all hyperparameters (model_name, prior, input_type,
                  hidden_size, input_size, number_components, z1_size, device, ...).
        """
        super(BaseModel, self).__init__()
        self.args = args
        # K>1 resampling is reserved for IWAE (future separate model).
        # Non-IWAE models always use a single latent sample.
        self.resample = False

        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()

        if self.args.prior == 'exemplar_prior':
            # Scalar learnable log-variance shared across all exemplar mixture components.
            # Using a single global variance keeps the prior well-regularised during early
            # training when exemplar embeddings are still random.
            self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))

        # ----- decoder output heads -----
        # The decoder architecture differs by input_type because the observation
        # likelihoods are fundamentally different distributions:
        #
        #   binary     — Bernoulli(p_x_mean): one sigmoid-squashed mean per pixel.
        #                No variance needed; the distribution is fully determined by mean.
        #
        #   gray /     — Either logistic-256 or diagonal Normal likelihood.
        #   continuous   Both need a per-pixel mean AND a variance.
        #                p_x_logvar: per-pixel log-variance (Hardtanh clamps to [-4.5, 0]
        #                  to prevent exploding/vanishing variance).
        #                decoder_logstd: a single global log-std scalar used by some
        #                  likelihood functions (e.g. log_logistic_256) as a temperature.
        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size))
            self.p_x_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size),
                                        activation=nn.Hardtanh(min_val=-4.5, max_val=0))
            self.decoder_logstd = torch.nn.Parameter(torch.tensor([0.], requires_grad=True))

        self.create_model(args)
        self.he_initializer()

    def he_initializer(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    @abstractmethod
    def create_model(self, args):
        pass

    @abstractmethod
    def kl_loss(self, latent_stats, exemplars_embeddin, dataset, cache, x_indices):
        pass

    def reconstruction_loss(self, x, x_mean, x_logvar):
        """Compute log p(x|z) — the reconstruction log-likelihood."""
        if self.args.input_type == 'binary':
            return log_bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            if self.args.use_logit is True:
                return log_normal_diag(x, x_mean, x_logvar, dim=1)
            else:
                return log_logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

    def calculate_loss(self, x, beta=1., average=False,
                       exemplars_embedding=None, cache=None, dataset=None):
        """Compute the (beta-)ELBO loss for a batch.

        ELBO = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))
        loss = -ELBO  (minimised by the optimiser)

        Args:
            x: Tuple of (data, x_indices) where data is (batch, D) and
               x_indices is (batch, 1) — indices into the training set used
               for exemplar prior leave-one-out masking.
            beta: KL weight for beta-VAE / annealing (default 1.0).
            average: If True, return scalar means; otherwise return (batch,) tensors.
            exemplars_embedding: Pre-computed prior mixture parameters, or None.
            cache: Cached encoder outputs for approximate nearest-neighbour search.
            dataset: Full training TensorDataset, needed when sampling exemplars on the fly.

        Returns:
            Tuple (loss, RE, KL) — each scalar if average=True else (batch,).
        """
        x, x_indices = x

        # ----- guards for unsupported K > 1 combinations -----
        if self.resample and self.args.prior == 'exemplar_prior':
            raise NotImplementedError(
                "K > 1 is not supported with exemplar_prior")
        if getattr(self.args, 'IW', False):
            raise ValueError(
                "The --IW flag is not used. For importance-weighted training, "
                "use --model_name iwae --K <num_samples> instead.")

        batch_size = x.size(0)
        x_mean, x_logvar, latent_stats = self.forward(x)
        # x_mean:  (batch, D)  or  (K*batch, D) when resample=True
        # x_logvar:(batch, D)  or  (K*batch, D) when resample=True

        # ----- align x with the K-expanded decoder outputs -----
        # reparameterize() already drew K samples, so x_mean has shape
        # (K*batch, D). We must tile the original x the same way so that
        # reconstruction_loss() receives matching tensors.
        if self.resample:
            x_expanded = (
                x.unsqueeze(0).expand(self.args.K, -1, -1)  # (K, batch, D)
                .contiguous().view(-1, x.size(-1))           # (K*batch, D)
            )
        else:
            x_expanded = x  # (batch, D)

        RE = self.reconstruction_loss(x_expanded, x_mean, x_logvar)   # log p(x|z), shape: (batch,) or (K*batch,)
        KL = self.kl_loss(latent_stats, exemplars_embedding, dataset, cache, x_indices)  # KL(q(z|x) || p(z)), shape: (batch,) or (K*batch,)

        # ----- multi-sample (K > 1) averaging -----
        # Fold the K dimension back so that each data point gets a single
        # gradient signal averaged across all K samples.
        if self.resample:
            RE = RE.view(self.args.K, batch_size).mean(dim=0)  # (batch,)
            KL = KL.view(self.args.K, batch_size).mean(dim=0)  # (batch,)

        loss = -RE + beta * KL   # -ELBO = -log p(x|z) + beta * KL(q||p), shape: (batch,)

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    def reparameterize(self, mu, logvar):
        """Sample z ~ N(mu, exp(0.5 * logvar)) via the reparameterization trick.

        Uses torch.distributions.Normal for clean, differentiable sampling.
        When K > 1, draws K independent samples per data point and flattens
        to (K * batch_size, z_dim).
        """
        z_std = torch.exp(0.5 * logvar)
        dist = torch.distributions.Normal(mu, z_std)

        if self.resample:
            batch_size = mu.size(0)
            z = dist.rsample([self.args.K])                # (K, batch, z_dim)
            z = z.view(self.args.K * batch_size, -1)       # (K*batch, z_dim)
        else:
            z = dist.rsample()                             # (batch, z_dim)
        return z

    def log_p_z_vampprior(self, z, exemplars_embedding):
        """Compute log p(z) under the VampPrior: a uniform mixture of C Gaussians.

        Each mixture component is a Gaussian whose mean and variance are the
        encoder's posterior parameters evaluated at a learnable pseudo-input.

        Args:
            z: Latent samples, shape (batch, z_dim).
            exemplars_embedding: Pre-computed (z_p_mean, z_p_logvar) tuple (C, z_dim),
                                 or None to compute from pseudo-inputs on the fly.

        Returns:
            Log-unnormalised mixture probabilities, shape (batch, C).
            Caller (log_p_z) applies logsumexp to get the final log-probability.
        """
        if exemplars_embedding is None:
            C = self.args.number_components
            # idle_input is a (C, C) identity matrix that selects each pseudo-input
            # via an embedding lookup implemented as a linear layer (self.means).
            X = self.means(self.idle_input)                  # (C, input_dim)
            z_p_mean, z_p_logvar = self.q_z(X, prior=True)  # (C, z_dim)
        else:
            C = torch.tensor(self.args.number_components).float()
            z_p_mean, z_p_logvar = exemplars_embedding       # each (C, z_dim)

        # Broadcast z against all C components without materialising a full replica.
        z_expand = z.unsqueeze(1)         # (batch, 1, z_dim)
        means = z_p_mean.unsqueeze(0)     # (1, C, z_dim)
        logvars = z_p_logvar.unsqueeze(0) # (1, C, z_dim)
        # log_normal_diag sums over dim=2 (z_dim), giving per-component log-probs.
        return log_normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # (batch, C)

    def log_p_z_exemplar(self, z, z_indices, exemplars_embedding, test):
        """Compute log p(z) under the exemplar prior with leave-one-out (LOO) masking.

        The exemplar prior is a uniform mixture over C exemplar Gaussians drawn from
        the training set. During training, each data point x_i must NOT appear as one
        of its own prior components — otherwise the model trivially memorises training
        data by collapsing q(z|x_i) onto the exemplar at index i. LOO masking removes
        self-matching exemplars from the mixture denominator and sets their log-prob
        to -inf so they contribute zero weight after exponentiation.

        Args:
            z: Latent samples, shape (batch, z_dim).
            z_indices: Training-set indices of the current batch, shape (batch, 1).
                       Used to identify which exemplars to mask out.
            exemplars_embedding: Tuple (centers, center_log_variance, center_indices):
                - centers: Exemplar means, shape (C, z_dim).
                - center_log_variance: Shared log-variance, shape (C, z_dim) but only
                  the first row is used (all rows are identical — same scalar broadcast).
                - center_indices: Training-set indices of the C exemplars, shape (C, 1).
            test: If True (or during eval mode), skip LOO masking — test queries should
                  be scored against the full mixture.

        Returns:
            Log-unnormalised mixture probabilities, shape (batch, C).
            Caller (log_p_z) applies logsumexp to get the final log-probability.
        """
        centers, center_log_variance, center_indices = exemplars_embedding
        # denominator tracks how many components are valid for each data point;
        # starts at C and gets decremented for each masked-out self-match.
        denominator = torch.tensor(len(centers)).expand(len(z)).float().to(self.args.device)  # (batch,)

        # All exemplars share the same global log-variance scalar (prior_log_variance).
        # center_log_variance has shape (C, z_dim); taking row 0 and broadcasting
        # avoids redundant memory — every row is the same repeated value.
        center_log_variance = center_log_variance[0, :].unsqueeze(0)  # (1, z_dim)

        prob, _ = log_normal_diag_vectorized(z, centers, center_log_variance)  # (batch, C)

        # ----- leave-one-out masking (training only) -----
        if test is False and self.args.no_mask is False:
            # Build a boolean mask of shape (batch, C) that is True wherever
            # a batch element's training index matches an exemplar's training index.
            # unsqueeze/expand broadcast the two index vectors into a comparison matrix.
            mask = z_indices.expand(-1, len(center_indices)) \
                    == center_indices.squeeze().unsqueeze(0).expand(len(z_indices), -1)
            # float('-inf') zeroes out self-matches after torch.exp(), preventing the
            # model from exploiting its own encoder embedding as a free prior shortcut.
            prob.masked_fill_(mask, value=float('-inf'))
            # Adjust denominator so the normalisation constant reflects only the
            # valid (unmasked) components for each data point.
            denominator = denominator - mask.sum(dim=1).float()  # (batch,)

        # Normalise by the number of active components (in log space).
        prob -= torch.log(denominator).unsqueeze(1)  # (batch, C)
        return prob

    # Each prior has its own helper because their logic is fundamentally different:
    #   vampprior   — mixture of C Gaussians from pseudo-inputs; also called from
    #                 generate_z() when sampling, so it lives as a standalone method
    #   exemplar    — leave-one-out masking with z_indices + train/test switching;
    #                 too stateful to fold into the dispatcher below
    # log_p_z() is the dispatcher that routes to the right helper and applies
    # the shared stable log-sum-exp over mixture components.
    def log_p_z(self, z, exemplars_embedding, sum=True, test=None):
        """Compute log p(z) under the configured prior.

        Args:
            z: Tuple of (latent_samples, x_indices) where latent_samples is (batch, z_dim)
               and x_indices is (batch, 1) used for LOO masking with exemplar prior.
            exemplars_embedding: Prior mixture parameters (format depends on prior type),
                                 or None for vampprior when using learnable pseudo-inputs.
            sum: If True, apply logsumexp over mixture components and return (batch,).
                 If False, return the raw per-component log-probs (batch, C) — used
                 by callers that need the full mixture for downstream computations.
            test: Override train/eval mode detection. None means infer from self.training.

        Returns:
            log_prior: shape (batch,) if sum=True, else (batch, C).
        """
        z, z_indices = z
        if test is None:
            # Infer from the module's own train/eval state so callers don't need
            # to pass test= explicitly during normal training/validation loops.
            test = not self.training
        if self.args.prior == 'standard':
            return log_normal_standard(z, dim=1)  # (batch,) — no mixture, skip logsumexp
        elif self.args.prior == 'vampprior':
            prob = self.log_p_z_vampprior(z, exemplars_embedding)  # (batch, C)
        elif self.args.prior == 'exemplar_prior':
            prob = self.log_p_z_exemplar(z, z_indices, exemplars_embedding, test)  # (batch, C)
        else:
            raise Exception('Wrong name of the prior!')

        if sum:
            # ----- numerically stable logsumexp over mixture components -----
            # Naive log(sum(exp(prob))) overflows/underflows with large C.
            # Subtracting prob_max before exponentiating keeps values in [0, 1],
            # then we add prob_max back to recover the true log-sum-exp.
            prob_max, _ = torch.max(prob, 1)              # (batch,)
            log_prior = prob_max + torch.log(
                torch.sum(torch.exp(prob - prob_max.unsqueeze(1)), 1)
            )                                             # (batch,)
        else:
            return prob  # (batch, C)
        return log_prior  # (batch,)

    def add_pseudoinputs(self):
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.means = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False, activation=nonlinearity)
        # init pseudo-inputs
        if self.args.use_training_data_init:
            self.means.linear.weight.data = self.args.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)
        self.idle_input = Variable(torch.eye(self.args.number_components, self.args.number_components), requires_grad=False)
        self.idle_input = self.idle_input.to(self.args.device)

    def generate_z_interpolate(self, exemplars_embedding=None, dim=0):
        new_zs = []
        exemplars_embedding, _, _ = exemplars_embedding
        step_counts = 10
        step = (exemplars_embedding[1] - exemplars_embedding[0])/step_counts
        for i in range(step_counts):
            new_z = exemplars_embedding[0].clone()
            new_z += i*step
            new_zs.append(new_z.unsqueeze(0))
        return torch.cat(new_zs, dim=0)

    def generate_z(self, N=25, dataset=None):
        """Sample N latent vectors from the prior p(z).

        Sampling strategy differs per prior:
          - standard: direct N(0, I) draw — no encoder needed.
          - vampprior: encode the first N pseudo-inputs, then reparameterise.
            Pseudo-inputs are sorted by their training-time index, so this
            consistently generates from the most-used mixture components.
          - exemplar_prior: pick N random training images, encode them with the
            prior encoder (fixed variance), then reparameterise. This mirrors
            the generative process: first draw an exemplar, then draw z given it.

        Args:
            N: Number of latent samples to generate.
            dataset: TensorDataset with training data. Required for exemplar_prior;
                     ignored for standard and vampprior.

        Returns:
            z_sample_rand: shape (N, z1_size), on self.args.device.
        """
        if self.args.prior == 'standard':
            # Simple ancestral sample from the isotropic Gaussian prior.
            z_sample_rand = torch.FloatTensor(N, self.args.z1_size).normal_().to(self.args.device)  # (N, z1_size)
        elif self.args.prior == 'vampprior':
            # Encode pseudo-inputs to get mixture component parameters,
            # then draw one sample from each of the first N components.
            means = self.means(self.idle_input)[0:N]                             # (N, input_dim)
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)             # each (N, z1_size)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)  # (N, z1_size)
            z_sample_rand = z_sample_rand.to(self.args.device)
        elif self.args.prior == 'exemplar_prior':
            # Ancestral sampling: (1) pick a random training exemplar, (2) encode it
            # with prior=True (uses learnable global variance instead of q_z_logvar),
            # (3) draw z from that Gaussian.
            rand_indices = torch.randint(low=0, high=self.args.training_set_size, size=(N,))
            exemplars = dataset.tensors[0][rand_indices]                         # (N, input_dim)
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(exemplars.to(self.args.device), prior=True)  # each (N, z1_size)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)  # (N, z1_size)
            z_sample_rand = z_sample_rand.to(self.args.device)
        return z_sample_rand

    def reference_based_generation_z(self, N=25, reference_image=None):
        pseudo, log_var = self.q_z(reference_image.to(self.args.device), prior=True)
        pseudo = pseudo.unsqueeze(1).expand(-1, N, -1).reshape(-1, pseudo.shape[-1])
        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, N, pseudo.shape[1])
        return z_sample_rand

    def reconstruct_x(self, x):
        x_reconstructed, _, z = self.forward(x)
        if self.args.model_name == 'pixelcnn':
            x_reconstructed = self.pixelcnn_generate(z[0].reshape(-1, self.args.z1_size), z[3].reshape(-1, self.args.z2_size))
        return x_reconstructed

    def logit_inverse(self, x):
        sigmoid = torch.nn.Sigmoid()
        lambd = self.args.lambd
        return ((sigmoid(x) - lambd)/(1-2*lambd))

    def generate_x(self, N=25, dataset=None):
        z2_sample_rand = self.generate_z(N=N, dataset=dataset)
        return self.generate_x_from_z(z2_sample_rand)

    def reference_based_generation_x(self, N=25, reference_image=None):
        z2_sample_rand = \
            self.reference_based_generation_z(N=N, reference_image=reference_image)
        generated_x = self.generate_x_from_z(z2_sample_rand)
        return generated_x

    def generate_x_interpolate(self, exemplars_embedding, dim=0):
        zs = self.generate_z_interpolate(exemplars_embedding, dim=dim)
        print(zs.shape)
        return self.generate_x_from_z(zs, with_reparameterize=False)

    def reshape_variance(self, variance, shape):
        return variance[0]*torch.ones(shape).to(self.args.device)

    def q_z(self, x, prior=False):
        if  'conv' in self.args.model_name or 'pixelcnn'==self.args.model_name:
            x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        h = self.q_z_layers(x)
        if self.args.model_name == 'convhvae_2level' or self.args.model_name=='pixelcnn':
            h = h.view(x.size(0), -1)
        z_q_mean = self.q_z_mean(h)
        if prior is True:
            if self.args.prior == 'exemplar_prior':
                z_q_logvar = self.prior_log_variance * torch.ones((x.shape[0], self.args.z1_size)).to(self.args.device)
                if self.args.model_name == 'newconvhvae_2level':
                    z_q_logvar = z_q_logvar.reshape(-1, 4, 4, 4)
            else:
                z_q_logvar = self.q_z_logvar(h)
        else:
            z_q_logvar = self.q_z_logvar(h)
        return z_q_mean.reshape(-1, self.args.z1_size), z_q_logvar.reshape(-1, self.args.z1_size)

    def cache_z(self, dataset, prior=True, cuda=True):
        cached_z = []
        cached_log_var = []
        caching_batch_size = 10000
        num_batchs = math.ceil(len(dataset) / caching_batch_size)
        for i in range(num_batchs):
            if len(dataset[0]) == 3:
                batch_data, batch_indices, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]
            else:
                batch_data, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]

            exemplars_embedding, log_variance_z = self.q_z(batch_data.to(self.args.device), prior=prior)
            cached_z.append(exemplars_embedding)
            cached_log_var.append(log_variance_z)
        cached_z = torch.cat(cached_z, dim=0)
        cached_log_var = torch.cat(cached_log_var, dim=0)
        cached_z = cached_z.to(self.args.device)
        cached_log_var = cached_log_var.to(self.args.device)
        return cached_z, cached_log_var

    def get_exemplar_set(self, z_mean, z_log_var, dataset, cache, x_indices):
        """Build the exemplar mixture that defines p(z) for the current batch.

        Two strategies are supported:

        1. Random (approximate_prior=False):
           Draw `number_components` random training images and encode them.
           This is unbiased but may produce a poor prior for any individual
           data point — unlikely to include the point's true nearest neighbours.

        2. Approximate nearest-neighbour (approximate_prior=True):
           Use cached encoder outputs to find approximate nearest neighbours
           in latent space, then build the mixture from those neighbours.
           Biased toward the batch's own neighbourhood, making the prior tighter
           and the training signal more informative.

        Args:
            z_mean:     Encoder means for the current batch, shape (batch, z_dim).
            z_log_var:  Encoder log-variances for the current batch, shape (batch, z_dim).
            dataset:    Full training TensorDataset (data is at dataset.tensors[0]).
            cache:      Tuple (cached_z, cached_log_var) of pre-encoded full dataset.
                        Only used when approximate_prior=True.
            x_indices:  Training-set indices for the current batch, shape (batch, 1).
                        Passed through for approximate KNN search and LOO masking.

        Returns:
            exemplar_set: Tuple (exemplars_z, log_variance, exemplars_indices):
                - exemplars_z:      shape (C, z_dim)
                - log_variance:     shape (C, z_dim)
                - exemplars_indices: shape (C,) — indices into the training set
        """
        if self.args.approximate_prior is False:
            # ----- random exemplar sampling -----
            exemplars_indices = torch.randint(low=0, high=self.args.training_set_size,
                                              size=(self.args.number_components, ))
            exemplars_z, log_variance = self.q_z(dataset.tensors[0][exemplars_indices].to(self.args.device), prior=True)
            # (number_components, z_dim) each
            exemplar_set = (exemplars_z, log_variance, exemplars_indices.to(self.args.device))
        else:
            # ----- approximate nearest-neighbour exemplar sampling -----
            exemplar_set = self.get_approximate_nearest_exemplars(
                z=(z_mean, z_log_var, x_indices),
                dataset=dataset,
                cache=cache)
        return exemplar_set

    def get_approximate_nearest_exemplars(self, z, cache, dataset):
        """Build an exemplar set from approximate nearest neighbours in latent space.

        Algorithm:
          1. Sample a large random candidate pool of size `number_components` from the
             full training set (sub-sampling avoids an O(N^2) pairwise search over N).
          2. Update the cache with the batch's fresh encoder outputs so the cache stays
             warm and converges toward accurate embeddings over training.
          3. Compute pairwise distances from each batch element to the candidate pool.
          4. Take the top-k nearest candidates per batch element, deduplicate, and
             re-encode those specific exemplars with the current (updated) encoder.
          5. Write the fresh embeddings back into the cache.

        The approximation is: step 1 restricts search to a random subset, not all N.
        This trades recall for O(number_components * batch) distance computation cost.

        Args:
            z: Tuple (z_mean, z_log_var, x_indices):
               - z_mean:    shape (batch, z_dim) — current batch latent means.
               - x_indices: shape (batch, 1)     — training-set indices of batch elements.
            cache: Tuple (cached_z, cached_log_var), each shape (N, z_dim).
                   Mutable — this method writes updated embeddings back in-place.
            dataset: Full training TensorDataset; data at dataset.tensors[0].

        Returns:
            exemplar_set: Tuple (exemplars_z, log_variance, exemplars_indices) — the
                          deduplicated nearest-neighbour exemplar embeddings and their
                          training-set indices.
        """
        # Step 1: random candidate pool of size number_components.
        exemplars_indices = torch.randint(low=0, high=self.args.training_set_size,
                                          size=(self.args.number_components, )).to(self.args.device)  # (C,)
        z, _, indices = z  # z: (batch, z_dim), indices: (batch, 1)
        cached_z, cached_log_variance = cache  # each (N, z_dim)

        # Step 2: write the current batch's fresh encoder means into the global cache
        # so future batches see an up-to-date embedding for these training points.
        cached_z[indices.reshape(-1)] = z  # in-place update; indices: (batch,)

        # Step 3: extract candidate embeddings and compute pairwise distances.
        sub_cache = cached_z[exemplars_indices, :]  # (C, z_dim)
        _, nearest_indices = pairwise_distance(z, sub_cache) \
            .topk(k=self.args.approximate_k, largest=False, dim=1)
        # nearest_indices: (batch, approximate_k) — local indices into sub_cache

        # Step 4: map local sub_cache indices back to global training-set indices,
        # deduplicate across batch elements, and re-encode with the current encoder.
        nearest_indices = torch.unique(nearest_indices.view(-1))  # (M,), M <= batch * approximate_k
        exemplars_indices = exemplars_indices[nearest_indices].view(-1)  # (M,) global indices
        # dataset lives on CPU; move indices there for indexing, then send data to device
        exemplars = dataset.tensors[0][exemplars_indices.cpu()].to(self.args.device)  # (M, input_dim)
        exemplars_z, log_variance = self.q_z(exemplars, prior=True)  # each (M, z_dim)

        # Step 5: write fresh embeddings for the selected exemplars back into the cache.
        cached_z[exemplars_indices] = exemplars_z

        exemplar_set = (exemplars_z, log_variance, exemplars_indices)
        return exemplar_set
