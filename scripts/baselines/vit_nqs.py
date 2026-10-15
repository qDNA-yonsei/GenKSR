#!/usr/bin/env python3
"""ViT-NQS variational baseline (paper Table 2 + Figure 9, Section 4.2).

Standalone port of the CANONICAL V2 notebook
    GenKSR/ViT_wave_function_V2.ipynb

It trains a translation-invariant Vision-Transformer neural quantum state
(ViT-NQS) on the 2D J1-J2 Heisenberg model (4x4 square lattice, PBC, J2=0.5)
via Variational Monte Carlo with Stochastic Reconfiguration (netket's VMC_SR
driver). It produces the "ViT-NQS" energy-per-site number reported in Table 2
and the energy-vs-iterations training curve of Figure 9.

------------------------------------------------------------------------------
Ported hyperparameters (from ViT_wave_function_V2.ipynb, cell 26/27)
------------------------------------------------------------------------------
Model (ViT ansatz):
    num_layers   = 4          # transformer encoder blocks
    d_model      = 64         # embedding dimension
    n_heads      = 4          # attention heads (factored multi-head attention)
    patch_size   = 2          # 2x2 patches -> n_patches = (L/patch)^2 = 4
    transl_invariant = True   # translation-invariant factored attention
    feed-forward = 4 * d_model (GELU), pre-LayerNorm encoder blocks
    output head  = LogCosh complex head (log_cosh of (real + i*imag) summed)
    param_dtype  = float64

Physics:
    Hypercube(length=L=4, n_dim=2, pbc=True, max_neighbor_order=2)
    Spin-1/2 Hilbert space with total_sz=0
    Heisenberg J1-J2: J=[1.0, J2=0.5], sign_rule=[False, False] (NO Marshall)
    -> .to_jax_operator()
    Reference (exact) ground-state energy for this 4x4, J2=0.5 cluster:
        E_exact = -33.831693  (notebook cell 28)

VMC / optimizer:
    sampler   = MetropolisExchange(d_max=2, n_chains=N_samples,
                                   sweep_size=lattice.n_nodes)
    n_samples = 4096, n_discard_per_chain=0, chunk_size=512
    optimizer = nk.optimizer.Sgd(learning_rate=0.0075)
    driver    = VMC_SR(diag_shift=1e-4, mode="complex")
    iterations (N_opt) in the notebook = 1500 (this script defaults to 800 per
               the CLI spec; pass --iterations 1500 to fully reproduce the run).

Number of parameters: the notebook's actual run reports N_params = 695488 for
these settings (d_model=64, 4 layers, FF = 4*d_model). NOTE: paper Table 5
quotes ~154780 params / 4 layers / d_model 64 / 4 heads / patch 2x2; the larger
count here comes from the wide FF (4*d_model) and the two-LayerNorm complex
output head in this implementation. The number printed by this script is the
ground truth for the ported code.

------------------------------------------------------------------------------
ViT notebook hyperparameter discrepancy
------------------------------------------------------------------------------
The OBSOLETE notebook ViT_wave_function.ipynb instantiates the ansatz as
    ViT(num_layers=4, d_model=60, n_heads=10, patch_size=2, transl_invariant=True)
whereas the CANONICAL V2 notebook uses
    ViT(num_layers=4, d_model=64, n_heads=4,  patch_size=2, transl_invariant=True)
Everything else (lattice, Hamiltonian, sampler, lr=0.0075, diag_shift=1e-4,
N_opt=1500, N_samples=4096) is identical between the two. This script follows
the V2 (d_model=64, n_heads=4) values, matching paper Table 5.

------------------------------------------------------------------------------
Run command (full reproduction of the V2 notebook)
------------------------------------------------------------------------------
    python scripts/baselines/vit_nqs.py \
        --j2 0.5 --L 4 --layers 4 --d-model 64 --n-heads 4 --patch 2 \
        --iterations 1500 --n-samples 4096 --lr 0.0075 \
        --output-dir figures/ --seed 0

Required optional dependencies (NOT installed by default):
    pip install netket flax optax
(jax is pulled in by netket; matplotlib is needed for the Figure 9 plot.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys


# -----------------------------------------------------------------------------
# Heavy / optional imports are wrapped so that a missing dependency produces a
# clear, actionable error message instead of a raw ImportError traceback.
# netket / flax / optax / jax are OPTIONAL deps and may not be installed.
# -----------------------------------------------------------------------------
def _import_heavy_deps():
    """Import netket/jax/flax, raising an actionable error if unavailable.

    Returns a namespace-like dict of the imported modules/objects.
    """
    missing = []
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except ImportError:
        missing.append("jax")
    try:
        import netket as nk  # noqa: F401
    except ImportError:
        missing.append("netket")
    try:
        import flax  # noqa: F401
        from flax import linen as nn  # noqa: F401
    except ImportError:
        missing.append("flax")
    try:
        import optax  # noqa: F401  (pulled in by netket's optimizers)
    except ImportError:
        missing.append("optax")
    try:
        from einops import rearrange  # noqa: F401
    except ImportError:
        missing.append("einops")

    if missing:
        raise ImportError(
            "ViT-NQS baseline requires optional dependencies that are not "
            "installed: " + ", ".join(sorted(set(missing))) + ".\n"
            "Install them with:\n\n"
            "    pip install netket flax optax einops\n\n"
            "(jax is pulled in automatically by netket; matplotlib is also "
            "needed for the Figure 9 plot.)"
        )

    import jax
    import jax.numpy as jnp
    import netket as nk
    import flax
    from flax import linen as nn
    from einops import rearrange

    return {
        "jax": jax,
        "jnp": jnp,
        "nk": nk,
        "flax": flax,
        "nn": nn,
        "rearrange": rearrange,
    }


# =============================================================================
# Flax ViT model definition.
#
# This is defined lazily inside a factory because the Flax `linen` module and
# the netket log_cosh activation must be imported first (they are optional
# dependencies). The classes below are verbatim ports of the V2 notebook.
# =============================================================================
def _build_model_classes(deps):
    jax = deps["jax"]
    jnp = deps["jnp"]
    nk = deps["nk"]
    nn = deps["nn"]
    rearrange = deps["rearrange"]

    from functools import partial

    # -- patch extraction (notebook cell 6) ----------------------------------
    def extract_patches2d(x, patch_size):
        batch = x.shape[0]
        n_patches = int((x.shape[1] // patch_size ** 2) ** 0.5)
        x = x.reshape(batch, n_patches, patch_size, n_patches, patch_size)
        x = x.transpose(0, 1, 3, 2, 4)
        x = x.reshape(batch, n_patches, n_patches, -1)
        x = x.reshape(batch, n_patches * n_patches, -1)
        return x

    # -- patch embedding (notebook cell 6) -----------------------------------
    class Embed(nn.Module):
        d_model: int  # dimensionality of the embedding space
        patch_size: int  # linear patch size
        param_dtype = jnp.float64

        def setup(self):
            self.embed = nn.Dense(
                self.d_model,
                kernel_init=nn.initializers.xavier_uniform(),
                param_dtype=self.param_dtype,
            )

        def __call__(self, x):
            x = extract_patches2d(x, self.patch_size)
            x = self.embed(x)
            return x

    # -- translation helper for translation-invariant attention (cell 12) ----
    @partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
    @partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
    def roll2d(spins, i, j):
        side = int(spins.shape[-1] ** 0.5)
        spins = spins.reshape(spins.shape[0], side, side)
        spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
        return spins.reshape(spins.shape[0], -1)

    # -- Factored Multi-Head Attention (notebook cell 12) --------------------
    class FMHA(nn.Module):
        d_model: int  # dimensionality of the embedding space
        n_heads: int  # number of heads
        n_patches: int  # length of the input sequence
        transl_invariant: bool = False
        param_dtype = jnp.float64

        def setup(self):
            self.v = nn.Dense(
                self.d_model,
                kernel_init=nn.initializers.xavier_uniform(),
                param_dtype=self.param_dtype,
            )
            self.W = nn.Dense(
                self.d_model,
                kernel_init=nn.initializers.xavier_uniform(),
                param_dtype=self.param_dtype,
            )
            if self.transl_invariant:
                self.alpha = self.param(
                    "alpha",
                    nn.initializers.xavier_uniform(),
                    (self.n_heads, self.n_patches),
                    self.param_dtype,
                )
                sq_n_patches = int(self.n_patches ** 0.5)
                assert sq_n_patches * sq_n_patches == self.n_patches
                self.alpha = roll2d(
                    self.alpha,
                    jnp.arange(sq_n_patches),
                    jnp.arange(sq_n_patches),
                )
                self.alpha = self.alpha.reshape(self.n_heads, -1, self.n_patches)
            else:
                self.alpha = self.param(
                    "alpha",
                    nn.initializers.xavier_uniform(),
                    (self.n_heads, self.n_patches, self.n_patches),
                    self.param_dtype,
                )

        def __call__(self, x):
            # apply the value matrix in parallel for each head
            v = self.v(x)

            # split the representations of the different heads
            v = rearrange(
                v,
                "batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff",
                n_heads=self.n_heads,
            )

            # factored attention mechanism
            v = rearrange(
                v, "batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff"
            )
            x = jnp.matmul(self.alpha, v)
            x = rearrange(
                x, "batch n_heads n_patches d_eff  -> batch n_patches n_heads d_eff"
            )

            # concatenate the different heads
            x = rearrange(
                x, "batch n_patches n_heads d_eff ->  batch n_patches (n_heads d_eff)"
            )

            # the representations of the different heads are combined together
            x = self.W(x)

            return x

    # -- Transformer encoder block (notebook cell 15) ------------------------
    class EncoderBlock(nn.Module):
        d_model: int  # dimensionality of the embedding space
        n_heads: int  # number of heads
        n_patches: int  # length of the input sequence
        transl_invariant: bool = False
        param_dtype = jnp.float64

        def setup(self):
            self.attn = FMHA(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                transl_invariant=self.transl_invariant,
            )

            self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
            self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)

            self.ff = nn.Sequential(
                [
                    nn.Dense(
                        4 * self.d_model,
                        kernel_init=nn.initializers.xavier_uniform(),
                        param_dtype=self.param_dtype,
                    ),
                    nn.gelu,
                    nn.Dense(
                        self.d_model,
                        kernel_init=nn.initializers.xavier_uniform(),
                        param_dtype=self.param_dtype,
                    ),
                ]
            )

        def __call__(self, x):
            x = x + self.attn(self.layer_norm_1(x))
            x = x + self.ff(self.layer_norm_2(x))
            return x

    # -- Transformer encoder (notebook cell 17) ------------------------------
    class Encoder(nn.Module):
        num_layers: int  # number of layers
        d_model: int  # dimensionality of the embedding space
        n_heads: int  # number of heads
        n_patches: int  # length of the input sequence
        transl_invariant: bool = False

        def setup(self):
            self.layers = [
                EncoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_patches=self.n_patches,
                    transl_invariant=self.transl_invariant,
                )
                for _ in range(self.num_layers)
            ]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    # -- LogCosh complex output head (notebook cell 21) ----------------------
    log_cosh = nk.nn.activation.log_cosh  # stable log(cosh(.))

    class OuputHead(nn.Module):  # noqa: N801 (verbatim notebook spelling)
        d_model: int  # dimensionality of the embedding space
        param_dtype = jnp.float64

        def setup(self):
            self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

            self.norm2 = nn.LayerNorm(
                use_scale=True, use_bias=True, param_dtype=self.param_dtype
            )
            self.norm3 = nn.LayerNorm(
                use_scale=True, use_bias=True, param_dtype=self.param_dtype
            )

            self.output_layer0 = nn.Dense(
                self.d_model,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=jax.nn.initializers.zeros,
            )
            self.output_layer1 = nn.Dense(
                self.d_model,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=jax.nn.initializers.zeros,
            )

        def __call__(self, x):
            z = self.out_layer_norm(x.sum(axis=1))

            out_real = self.norm2(self.output_layer0(z))
            out_imag = self.norm3(self.output_layer1(z))

            out = out_real + 1j * out_imag

            return jnp.sum(log_cosh(out), axis=-1)

    # -- full ViT ansatz (notebook cell 23) ----------------------------------
    class ViT(nn.Module):
        num_layers: int  # number of layers
        d_model: int  # dimensionality of the embedding space
        n_heads: int  # number of heads
        patch_size: int  # linear patch size
        transl_invariant: bool = False

        @nn.compact
        def __call__(self, spins):
            x = jnp.atleast_2d(spins)

            Ns = x.shape[-1]  # number of sites
            n_patches = Ns // self.patch_size ** 2  # length of the input sequence

            x = Embed(d_model=self.d_model, patch_size=self.patch_size)(x)

            y = Encoder(
                num_layers=self.num_layers,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=n_patches,
                transl_invariant=self.transl_invariant,
            )(x)

            log_psi = OuputHead(d_model=self.d_model)(y)

            return log_psi

    return ViT


# =============================================================================
# Argument parsing.
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "ViT-NQS variational baseline for the 2D J1-J2 Heisenberg model "
            "(paper Table 2 + Figure 9). Ported from ViT_wave_function_V2.ipynb."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Physics
    p.add_argument("--j2", type=float, default=0.5,
                   help="Next-nearest-neighbour coupling J2 (J1 fixed at 1.0).")
    p.add_argument("--L", type=int, default=4,
                   help="Linear size of the square lattice (LxL sites).")
    # ViT model (V2 notebook / Table 5 defaults)
    p.add_argument("--layers", type=int, default=4,
                   help="Number of transformer encoder layers.")
    p.add_argument("--d-model", type=int, default=64,
                   help="Embedding dimension d_model.")
    p.add_argument("--n-heads", type=int, default=4,
                   help="Number of (factored) attention heads.")
    p.add_argument("--patch", type=int, default=2,
                   help="Linear patch size (patch x patch).")
    # VMC / optimizer
    p.add_argument("--iterations", type=int, default=800,
                   help="Number of VMC optimization steps (notebook used 1500).")
    p.add_argument("--n-samples", type=int, default=4096,
                   help="Number of Monte Carlo samples / chains per step.")
    p.add_argument("--lr", type=float, default=0.0075,
                   help="SGD learning rate for the SR update.")
    p.add_argument("--diag-shift", type=float, default=1e-4,
                   help="Diagonal shift (regularization) for SR.")
    p.add_argument("--chunk-size", type=int, default=512,
                   help="Chunk size for the variational state evaluation.")
    # IO / misc
    p.add_argument("--output-dir", type=str, default="figures/",
                   help="Directory for the Figure 9 plot and energy JSON.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed.")
    p.add_argument("--no-transl-invariant", action="store_true",
                   help="Disable translation-invariant attention "
                        "(notebook uses translation invariance = ON).")
    return p.parse_args(argv)


# =============================================================================
# Main.
# =============================================================================
def main(argv=None):
    args = parse_args(argv)

    # Heavy imports first so dependency problems surface immediately with a
    # clear message (before doing any other work).
    deps = _import_heavy_deps()
    jax = deps["jax"]
    jnp = deps["jnp"]
    nk = deps["nk"]

    # float64 is required: the notebook uses param_dtype=float64 throughout.
    jax.config.update("jax_enable_x64", True)

    ViT = _build_model_classes(deps)

    transl_invariant = not args.no_transl_invariant

    L = args.L
    n_dim = 2
    J2 = args.j2

    print(f"[ViT-NQS] devices: {jax.devices()}", flush=True)
    print(
        f"[ViT-NQS] L={L} (n_dim={n_dim}, PBC), J=[1.0, {J2}], "
        f"layers={args.layers}, d_model={args.d_model}, n_heads={args.n_heads}, "
        f"patch={args.patch}, transl_invariant={transl_invariant}",
        flush=True,
    )

    key = jax.random.key(args.seed)

    # ----- lattice / Hilbert / Hamiltonian (notebook cell 26) ---------------
    lattice = nk.graph.Hypercube(
        length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2
    )

    hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

    # Heisenberg J1-J2 spin Hamiltonian, NO Marshall sign rule.
    hamiltonian = nk.operator.Heisenberg(
        hilbert=hilbert,
        graph=lattice,
        J=[1.0, J2],
        sign_rule=[False, False],
    ).to_jax_operator()

    # ----- variational wave function ----------------------------------------
    vit_module = ViT(
        num_layers=args.layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        patch_size=args.patch,
        transl_invariant=transl_invariant,
    )

    # Dummy spin configurations to initialize parameters (shape (M, L*L)).
    key, subkey = jax.random.split(key)
    spin_configs = (
        jax.random.randint(subkey, shape=(2, L * L), minval=0, maxval=1) * 2 - 1
    )

    key, subkey = jax.random.split(key)
    params = vit_module.init(subkey, spin_configs)

    # ----- sampler (notebook cell 26) ---------------------------------------
    N_samples = args.n_samples
    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=2,
        n_chains=N_samples,
        sweep_size=lattice.n_nodes,
    )

    # ----- optimizer (plain SGD; SR provides the metric/preconditioner) -----
    optimizer = nk.optimizer.Sgd(learning_rate=args.lr)

    # ----- variational Monte Carlo state ------------------------------------
    key, subkey = jax.random.split(key, 2)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=vit_module,
        sampler_seed=subkey,
        n_samples=N_samples,
        n_discard_per_chain=0,
        variables=params,
        chunk_size=args.chunk_size,
    )

    N_params = nk.jax.tree_size(vstate.parameters)
    print(f"[ViT-NQS] Number of parameters = {N_params}", flush=True)

    # ----- VMC_SR driver (notebook cell 26) ---------------------------------
    from netket.driver import VMC_SR

    vmc = VMC_SR(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        diag_shift=args.diag_shift,
        variational_state=vstate,
        mode="complex",
    )

    # ----- optimization loop (notebook cell 27) -----------------------------
    log = nk.logging.RuntimeLog()
    N_opt = args.iterations
    print(f"[ViT-NQS] running VMC_SR for {N_opt} iterations ...", flush=True)
    vmc.run(n_iter=N_opt, out=log)

    # ----- energy-per-site logging (notebook cell 30) -----------------------
    # NOTE: the V2 notebook divides the mean energy by (L * L * 4). This is the
    # exact normalization used to produce the Figure 9 curve and the reported
    # "Last value" (-0.4419...), so we replicate it verbatim. The raw total
    # energy mean is also reported for completeness / Table 2 cross-checking.
    import numpy as np

    energy_mean = np.asarray(log.data["Energy"]["Mean"]).real
    energy_per_site = energy_mean / (L * L * 4)

    final_energy_total = float(energy_mean[-1])
    final_energy_per_site = float(energy_per_site[-1])

    # Best (lowest) variational energy over the run, a more robust estimate of
    # the converged value than the last noisy sample.
    best_energy_total = float(np.min(energy_mean))
    best_energy_per_site = float(np.min(energy_per_site))

    print(f"[ViT-NQS] Final energy (total)    = {final_energy_total:.6f}", flush=True)
    print(f"[ViT-NQS] Final energy-per-site   = {final_energy_per_site:.8f}", flush=True)
    print(f"[ViT-NQS] Best  energy (total)    = {best_energy_total:.6f}", flush=True)
    print(f"[ViT-NQS] Best  energy-per-site   = {best_energy_per_site:.8f}", flush=True)

    # ----- outputs ----------------------------------------------------------
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Energy JSON.
    energy_json_path = os.path.join(output_dir, "vit_nqs_energy.json")
    result = {
        "model": "ViT-NQS",
        "source_notebook": "ViT_wave_function_V2.ipynb",
        "L": L,
        "n_dim": n_dim,
        "J": [1.0, J2],
        "j2": J2,
        "sign_rule": [False, False],
        "max_neighbor_order": 2,
        "num_layers": args.layers,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "patch_size": args.patch,
        "transl_invariant": transl_invariant,
        "n_params": int(N_params),
        "n_samples": N_samples,
        "learning_rate": args.lr,
        "diag_shift": args.diag_shift,
        "iterations": N_opt,
        "seed": args.seed,
        "energy_normalization": "Energy.Mean.real / (L*L*4)",
        "final_energy_total": final_energy_total,
        "final_energy_per_site": final_energy_per_site,
        "best_energy_total": best_energy_total,
        "best_energy_per_site": best_energy_per_site,
        # Exact reference for the 4x4, J2=0.5 cluster (notebook cell 28).
        "exact_energy_total": -33.831693,
        "exact_energy_per_site": -33.831693 / (L * L * 4),
        "energy_per_site_history": energy_per_site.tolist(),
        "energy_total_history": energy_mean.tolist(),
    }
    with open(energy_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ViT-NQS] saved energy JSON -> {energy_json_path}", flush=True)

    # Figure 9 plot (energy per site vs iterations).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(energy_per_site, lw=1.2, label="ViT-NQS")

        exact_per_site = -33.831693 / (L * L * 4)
        if abs(J2 - 0.5) < 1e-9 and L == 4:
            ax.axhline(
                exact_per_site,
                color="k",
                ls="--",
                lw=1.0,
                label=f"exact ({exact_per_site:.5f})",
            )

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Energy per site")
        ax.set_title(
            f"Figure 9: ViT-NQS training (J1-J2, L={L}, J2={J2})"
        )
        ax.legend()
        fig.tight_layout()

        png_path = os.path.join(output_dir, "fig9_vit_nqs_training.png")
        pdf_path = os.path.join(output_dir, "fig9_vit_nqs_training.pdf")
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"[ViT-NQS] saved Figure 9 -> {png_path}", flush=True)
        print(f"[ViT-NQS] saved Figure 9 -> {pdf_path}", flush=True)
    except ImportError:
        print(
            "[ViT-NQS] WARNING: matplotlib not available; skipped Figure 9 plot. "
            "Install with `pip install matplotlib`.",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"[ViT-NQS] DONE. Table 2 'ViT-NQS' energy-per-site = "
        f"{final_energy_per_site:.8f} (best {best_energy_per_site:.8f})",
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
