"""
Normalizing-flow jump proposal using the coppuccino package.

A copula normalizing flow (https://github.com/AaronDJohnson/coppuccino) is fit
to a thinned buffer of chain samples and used as an independence proposal.
The flow is retrained periodically after a user-defined number of sampler
iterations, so the proposal adapts to the posterior as the chain explores it.

The coppuccino package is an optional dependency of enterprise_extensions::

    pip install enterprise_extensions[nf]

Example usage with a PTMCMCSampler sampler (e.g. from setup_sampler)::

    from enterprise_extensions.sampler import setup_sampler
    from enterprise_extensions.nfjump import NFJump

    sampler = setup_sampler(pta, outdir="./chains")

    ndim = len(pta.param_names)
    nf = NFJump(ndim, train_interval=25000, prior_bounds=bounds)
    # optionally seed the proposal with samples from a previous run
    # nf = NFJump(ndim, train_interval=25000, initial_samples=old_chain)
    sampler.addProposalToCycle(nf, 20)

    sampler.sample(p0, Niter)

When running with MPI, each chain (temperature) carries its own NFJump
instance and trains its own flow, so the proposal matches the tempered
distribution it is sampling.

Aaron D. Johnson
"""

import numpy as np

try:
    from coppuccino import log_prob, normalizing_flows_fit, sample_and_log_prob, save_flow

    HAS_COPPUCCINO = True
except ImportError:
    HAS_COPPUCCINO = False


class NFJump(object):
    """Adaptive independence jump proposal based on a copula normalizing flow.

    Chain samples are accumulated in an internal buffer every time the jump is
    called. The buffer is automatically thinned (by keeping every other sample
    and doubling the recording stride) so it never exceeds ``max_samples``.
    Every ``train_interval`` sampler iterations a fresh flow is fit to the
    buffer, and proposals are then drawn from the flow with the appropriate
    Metropolis-Hastings correction qxy = log q(x) - log q(y).

    Until the first flow has been trained the proposal returns the current
    point unchanged (a null jump), so it is safe to include in the proposal
    cycle from the start of sampling. To make the proposal useful immediately,
    pass posterior samples from a previous run via ``initial_samples``.
    """

    def __init__(
        self,
        ndim,
        train_interval=25000,
        max_samples=10000,
        min_samples=1000,
        nburn=0,
        initial_samples=None,
        param_idx=None,
        prior_bounds=None,
        batch_size=256,
        rng_seed=999,
        save_file=None,
        fit_kwargs=None,
    ):
        """Initialize the normalizing-flow jump.

        :param ndim:            Total number of search parameters
        :param train_interval:  Number of sampler iterations between flow (re)trainings
        :param max_samples:     Maximum size of the training buffer; the buffer is
                                thinned to stay at or below this (default 10000)
        :param min_samples:     Minimum number of buffered samples before the first
                                training is attempted
        :param nburn:           Sampler iteration before which no samples are recorded
                                and no training is performed
        :param initial_samples: Optional array of posterior samples, shape (n, ndim)
                                or (n, len(param_idx)), used to train the flow at
                                construction time
        :param param_idx:       Optional list of parameter indices to model with the
                                flow; remaining parameters are left untouched by the
                                jump (default: all parameters)
        :param prior_bounds:    Optional array of shape (len(param_idx), 2) passed to
                                coppuccino so the marginal CDFs cover the full prior
                                support (recommended)
        :param batch_size:      Number of proposals drawn from the flow per call to
                                coppuccino (drawn in batches for efficiency)
        :param rng_seed:        Seed for the random number generator used for
                                training and sampling seeds
        :param save_file:       Optional path; if given, the flow is saved there
                                after every training
        :param fit_kwargs:      Optional dict of extra keyword arguments passed to
                                coppuccino.normalizing_flows_fit (e.g. max_epochs)
        """

        if not HAS_COPPUCCINO:
            raise ImportError(
                "The coppuccino package is required for NFJump. "
                "Install it with `pip install coppuccino` or `pip install enterprise_extensions[nf]`."
            )

        self.ndim = ndim
        self.train_interval = train_interval
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.nburn = nburn
        self.param_idx = np.arange(ndim) if param_idx is None else np.atleast_1d(param_idx)
        self.prior_bounds = prior_bounds
        self.batch_size = batch_size
        self.save_file = save_file
        self.fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)

        self.name = "NFJump"

        self.flow = None
        self._rng = np.random.default_rng(rng_seed)
        self._buffer = []
        self._record_every = 1
        self._call_count = 0
        self._last_train_iter = 0
        self._proposal_cache = None
        self._cache_idx = 0

        if initial_samples is not None:
            self.add_samples(initial_samples)
            self.train()

    @property
    def __name__(self):
        return self.name

    @property
    def n_buffered(self):
        """Number of samples currently in the training buffer."""
        return len(self._buffer)

    def add_samples(self, samples):
        """Feed samples into the training buffer (e.g. from a previous run).

        :param samples: array of shape (n, ndim) or (n, len(param_idx))
        """
        samples = np.atleast_2d(np.asarray(samples, dtype=float))

        if samples.shape[1] == self.ndim:
            samples = samples[:, self.param_idx]
        elif samples.shape[1] != len(self.param_idx):
            raise ValueError(
                "Samples have %d columns; expected %d (ndim) or %d (len(param_idx))"
                % (samples.shape[1], self.ndim, len(self.param_idx))
            )

        # thin the incoming samples so they fit within the buffer cap
        if samples.shape[0] > self.max_samples:
            idx = np.linspace(0, samples.shape[0] - 1, self.max_samples).astype(int)
            samples = samples[idx]

        self._buffer.extend(samples)
        self._cap_buffer()

    def train(self, iter=0):
        """Fit the normalizing flow to the current buffer.

        Called automatically every ``train_interval`` iterations; can also be
        called by hand. On failure the previous flow (if any) is kept.
        """
        data = np.asarray(self._buffer)
        seed = int(self._rng.integers(2**31 - 1))

        try:
            self.flow = normalizing_flows_fit(data, rng_seed=seed, prior_bounds=self.prior_bounds, **self.fit_kwargs)
        except Exception as e:
            print("WARNING: NFJump flow training failed (%s). Keeping previous flow." % str(e))
        else:
            self._proposal_cache = None
            if self.save_file is not None:
                save_flow(self.flow, self.save_file)

        self._last_train_iter = iter

    def _cap_buffer(self):
        """Keep the buffer at or below max_samples by thinning by a factor of 2."""
        while len(self._buffer) > self.max_samples:
            self._buffer = self._buffer[::2]
            self._record_every *= 2

    def _record(self, x, iter):
        """Record the current point into the training buffer (with thinning)."""
        if iter < self.nburn:
            return

        self._call_count += 1
        if self._call_count % self._record_every == 0:
            self._buffer.append(np.asarray(x, dtype=float)[self.param_idx].copy())
            self._cap_buffer()

    def _maybe_train(self, iter):
        """Retrain the flow if enough samples and iterations have accumulated."""
        if iter < self.nburn or len(self._buffer) < self.min_samples:
            return

        if self.flow is None or iter - self._last_train_iter >= self.train_interval:
            self.train(iter)

    def _next_proposal(self):
        """Return one (sample, log_prob) draw from the flow, batched for speed."""
        if self._proposal_cache is None or self._cache_idx >= self.batch_size:
            seed = int(self._rng.integers(2**31 - 1))
            self._proposal_cache = sample_and_log_prob(self.flow, self.batch_size, rng_seed=seed)
            self._cache_idx = 0

        draws, logps = self._proposal_cache
        draw, lp = draws[self._cache_idx], logps[self._cache_idx]
        self._cache_idx += 1

        return draw, float(lp)

    def __call__(self, x, iter, beta):
        """Propose a new point by drawing from the fitted flow.

        @param x: parameter vector at current position
        @param iter: sampler iteration number
        @param beta: inverse temperature of the chain

        @return: q: proposed parameter vector
        @return: qxy: forward-backward jump probability, log[q(x)/q(y)]
        """
        self._record(x, iter)
        self._maybe_train(iter)

        q = np.array(x, dtype=float)

        # no flow trained yet: null jump
        if self.flow is None:
            return q, 0.0

        draw, lp_q = self._next_proposal()
        lp_x = float(log_prob(self.flow, np.atleast_2d(q[self.param_idx]))[0])

        q[self.param_idx] = draw
        qxy = lp_x - lp_q

        # if the current point lies outside the flow's support, force rejection
        if not np.isfinite(qxy):
            qxy = -np.inf

        return q, qxy
