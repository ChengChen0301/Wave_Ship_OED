"""
GO OED with a Normalizing-Flow lower bound for utility.

Adapted for the current Ship_OED repository:
- Uses existing OEDCOMMON/GO functions for prior, forward/predictive models, and design generation.
- Computes a variational lower bound
    E_{theta,y}[ log q_phi(z|y,d) - log p(z) ]
  where q_phi is learned by a conditional normalizing flow.
"""

import argparse
import os, sys
import re
import time
from types import SimpleNamespace
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from OEDCOMMON.OED_common import (
    generate_design_candidates,
    get_LA_and_scores,
    get_surrogate_model_and_scaler,
    sample_theta_prior_6d,
)
from OEDCOMMON.observeModel import funcObs_GO
from OEDCOMMON.predictModel import funcPred_GO


def rs_tag_from_model_path(model_path):
    if not model_path:
        return ""
    match = re.search(r"(rs[0-9]+p[0-9]+)", os.path.basename(str(model_path)))
    return match.group(1) if match else ""


class FCNN(nn.Module):
    def __init__(self, in_dim, out_dim, nodes, act):
        super().__init__()
        layers = [nn.Linear(in_dim, nodes[0]), act]
        for i in range(len(nodes) - 1):
            layers += [nn.Linear(nodes[i], nodes[i + 1]), act]
        layers += [nn.Linear(nodes[-1], out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ConditionalAffine1D(nn.Module):
    """
    1D conditional affine: z' = z * exp(s(y)) + t(y).
    Scale s is bounded to stabilize exp(s) and log|det J|.
    """

    def __init__(self, hidden, s_scale: float = 0.35):
        super().__init__()
        self.s_scale = s_scale
        self.s = FCNN(1, 1, hidden, nn.ReLU())
        self.t = FCNN(1, 1, hidden, nn.ReLU())

    def forward(self, z, y):
        s = self.s_scale * torch.tanh(self.s(y))
        t = self.t(y)
        zt = z * torch.exp(s) + t
        log_det = s[:, 0]
        return zt, log_det


class ConditionalTanhWarp1D(nn.Module):
    """
    Monotone nonlinear 1D transform conditioned on y:
        z' = z + c(y) * tanh(z)
    with c(y) in (-c_max, c_max), c_max < 1 to ensure positive Jacobian:
        dz'/dz = 1 + c(y) * (1 - tanh(z)^2) >= 1 - c_max > 0
    """

    def __init__(self, hidden, c_max: float = 0.85):
        super().__init__()
        self.c_max = float(min(max(c_max, 1e-3), 0.99))
        self.c = FCNN(1, 1, hidden, nn.ReLU())

    def forward(self, z, y):
        c = self.c_max * torch.tanh(self.c(y))
        tanh_z = torch.tanh(z)
        zt = z + c * tanh_z
        jac = 1.0 + c * (1.0 - tanh_z * tanh_z)
        log_det = torch.log(torch.clamp(jac, min=1e-8))[:, 0]
        return zt, log_det


class FlowModel(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z, y):
        log_det = torch.zeros(z.shape[0], device=z.device)
        for f in self.flows:
            z, ld = f.forward(z, y)
            log_det += ld
        return z, log_det



def _roll_threshold(args):
    threshold = getattr(args, "threshold", None)
    if threshold is None:
        tag = rs_tag_from_model_path(args.surrogate_forward_model_path)
        threshold = float(tag[2:].replace("p", ".")) if tag else 0.25
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("--threshold must be finite and positive")
    return float(threshold)


def _threshold_tag(args):
    return "rs" + format(_roll_threshold(args), "g").replace(".", "p")


def _prepare_models_and_wrappers(args):
    threshold = _roll_threshold(args)
    for enabled, path in (
        (args.use_surrogate_forward, args.surrogate_forward_model_path),
        (args.use_surrogate_predict, args.surrogate_predict_model_path),
    ):
        tag = rs_tag_from_model_path(path)
        if enabled and tag and not np.isclose(float(tag[2:].replace("p", ".")), threshold):
            raise ValueError(f"Surrogate {path} does not match threshold {threshold}")
    forward_model, forward_scaler = get_surrogate_model_and_scaler(
        args.surrogate_forward_model_path,
        args.surrogate_forward_scaler_path,
        device=args.device,
    )
    predict_model, predict_scaler = get_surrogate_model_and_scaler(
        args.surrogate_predict_model_path,
        args.surrogate_predict_scaler_path,
        device=args.device,
    )
    la_samples, scores = get_LA_and_scores(n_groups=args.nGroupsLA, seed=args.seed)

    use_sf = bool(args.use_surrogate_forward)
    use_sp = bool(args.use_surrogate_predict)
    fm_obs = forward_model if use_sf else None
    fs_obs = forward_scaler if use_sf else None

    def obs_fn(theta, d):
        return funcObs_GO(theta, d, use_sf, fm_obs, fs_obs, device=args.device, threshold=threshold)

    pred_fn = partial(
        funcPred_GO,
        forward_surrogate_model=forward_model,
        forward_surrogate_scaler=forward_scaler,
        predict_surrogate_model=predict_model,
        predict_surrogate_scaler=predict_scaler,
        LA_samples=la_samples,
        scores=scores,
        device=args.device,
        use_forward_surrogate=use_sf and not use_sp,
        use_predict_surrogate=use_sp,
        threshold=threshold,
    )
    return obs_fn, pred_fn, forward_model, forward_scaler


def _fit_hurdle_log_kde_prior(z_values, *, bw_lo=-3.0, bw_hi=0.0, n_grid=120, eps0=1e-12):
    z = np.asarray(z_values, dtype=float).reshape(-1)
    zero = np.isclose(z, 0.0, atol=float(eps0))
    z_pos = z[(~zero) & (z > float(eps0))]
    p0 = float(np.mean(zero))
    tiny = np.finfo(float).tiny
    model = {"p0": p0, "eps0": float(eps0), "kde": None}
    log_pz = np.full(z.shape, np.log(tiny), dtype=float)
    log_pz[zero] = np.log(max(p0, tiny))
    if z_pos.size <= 5:
        return log_pz, model

    v_pos = np.log(z_pos).reshape(-1, 1)
    grid = GridSearchCV(
        estimator=KernelDensity(kernel="gaussian"),
        param_grid={"bandwidth": 10 ** np.linspace(float(bw_lo), float(bw_hi), int(n_grid))},
    )
    grid.fit(v_pos)
    model["kde"] = grid.best_estimator_

    pos_eval = (~zero) & (z > float(eps0))
    if np.any(pos_eval):
        v_eval = np.log(z[pos_eval]).reshape(-1, 1)
        log_pz[pos_eval] = (
            np.log(max(1.0 - p0, tiny))
            + model["kde"].score_samples(v_eval)
            - np.log(z[pos_eval])
        )
    return log_pz, model


def _hurdle_log_kde_logpdf(z_values, model):
    z = np.asarray(z_values, dtype=float).reshape(-1)
    p0 = float(model["p0"])
    eps0 = float(model.get("eps0", 1e-12))
    kde = model.get("kde")
    tiny = np.finfo(float).tiny
    zero = np.isclose(z, 0.0, atol=eps0)
    log_pz = np.full(z.shape, np.log(tiny), dtype=float)
    log_pz[zero] = np.log(max(p0, tiny))
    pos = (~zero) & (z > eps0)
    if np.any(pos) and kde is not None:
        log_pz[pos] = (
            np.log(max(1.0 - p0, tiny))
            + kde.score_samples(np.log(z[pos]).reshape(-1, 1))
            - np.log(z[pos])
        )
    return log_pz


def _torch_scaler_transform(x, scaler, device):
    mean = torch.as_tensor(np.asarray(scaler.mean_, dtype=np.float32), device=device)
    scale = torch.as_tensor(np.asarray(scaler.scale_, dtype=np.float32), device=device)
    return (x - mean) / torch.clamp(scale, min=1e-12)


def _torch_obs_surrogate(theta_t, d_t, model, scaler, device):
    """Differentiable observe surrogate h(theta, d) for continuous design optimization."""
    if theta_t.ndim != 2 or theta_t.shape[1] != 6:
        raise ValueError(f"theta_t must have shape (n, 6); got {tuple(theta_t.shape)}")
    d_row = d_t.reshape(1, 2).expand(theta_t.shape[0], 2)
    x = torch.cat([d_row, theta_t], dim=1)
    x_s = _torch_scaler_transform(x, scaler, device)
    model.eval()
    out = model(x_s)
    if isinstance(out, tuple):
        raise ValueError("The observe surrogate must be scalar-output; got a tuple-output model.")
    out = out.reshape(-1)
    y_mean = getattr(scaler, "y_mean_", None)
    y_scale = getattr(scaler, "y_scale_", None)
    if y_mean is not None and y_scale is not None:
        ym = torch.as_tensor(np.asarray(y_mean, dtype=np.float32).ravel()[0], device=device)
        ys = torch.as_tensor(np.asarray(y_scale, dtype=np.float32).ravel()[0], device=device)
        out = out * ys + ym
        if str(getattr(scaler, "target_transform_", "")).lower() == "log":
            out = torch.exp(out)
    return out.reshape(-1, 1)


def _design_bounds_from_args(args):
    tp = float(getattr(args, "Tp", 15.0))
    hs = float(getattr(args, "Hs", 12.0))
    l_lo = float(getattr(args, "L_Tp_min", 0.1)) * tp
    l_hi = float(getattr(args, "L_Tp_max", 6.0)) * tp
    a_lo = float(getattr(args, "A_Hs_min", 0.0)) * hs
    a_hi = float(getattr(args, "A_Hs_max", 1.5)) * hs
    return np.array([l_lo, a_lo], dtype=np.float32), np.array([l_hi, a_hi], dtype=np.float32)


def _separated_logq(pi_net, flow_model, base, y_batch, z_s_batch, zero_batch, log_dt_dz_batch, t_sd):
    eps = 1e-8
    pi = torch.sigmoid(pi_net(y_batch)).squeeze(-1)
    logq = zero_batch * torch.log(pi + eps)
    pos = zero_batch < 0.5
    if torch.any(pos):
        zf, ld = flow_model.forward(z_s_batch[pos], y_batch[pos])
        log_prob_pos = base.log_prob(zf[:, 0]) + ld
        log_prob_pos = log_prob_pos - np.log(max(float(t_sd), 1e-12)) + log_dt_dz_batch[pos]
        logq[pos] = torch.log(1.0 - pi[pos] + eps) + log_prob_pos
    return logq


def estimate_nf_lower_bound(
    design,
    obs_fn,
    pred_fn,
    args,
    *,
    return_training_history: bool = False,
):
    rng = np.random.default_rng(args.seed)
    theta = sample_theta_prior_6d(args.nout, seed=args.seed)
    g = obs_fn(theta, design).reshape(-1, 1)
    noise = rng.normal(0.0, args.sigma_noise, size=(theta.shape[0], 1))
    y = g + noise
    z = pred_fn(theta).reshape(-1, 1)
    eps0 = float(getattr(args, "eps0", 1e-12))
    bad = (z[:, 0] < -eps0)
    if np.any(bad):
        raise ValueError(
            "separated + log NF requires nonnegative z values; "
            f"got {int(np.sum(bad))} values below -eps0."
        )
    log_pz, prior_model = _fit_hurdle_log_kde_prior(z[:, 0], eps0=eps0)

    zero_mask = np.isclose(z[:, 0], 0.0, atol=eps0)
    pos_mask = (~zero_mask) & (z[:, 0] > eps0)
    if not np.any(pos_mask):
        raise ValueError("separated + log NF requires at least one positive z sample.")

    t_pos = np.log(z[pos_mask, 0]).reshape(-1, 1)
    t_mu = float(np.mean(t_pos))
    t_sd = max(float(np.std(t_pos)), 1e-12)
    z_s = np.zeros_like(z, dtype=np.float32)
    z_s[pos_mask, 0] = ((t_pos[:, 0] - t_mu) / t_sd).astype(np.float32)
    log_dt_dz = np.zeros(z.shape[0], dtype=float)
    log_dt_dz[pos_mask] = -np.log(z[pos_mask, 0])

    y_mu = float(np.mean(y))
    y_sd = max(float(np.std(y)), 1e-12)
    y_s = ((y - y_mu) / y_sd).astype(np.float32)

    n = z.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.8 * n)
    tr, te = idx[:n_train], idx[n_train:]

    # torch.manual_seed(args.seed)
    train_seed = 42
    torch.manual_seed(train_seed)

    device = torch.device(args.device)
    y_t = torch.tensor(y_s, dtype=torch.float32, device=device)
    z_s_t = torch.tensor(z_s, dtype=torch.float32, device=device)
    zero_t = torch.tensor(zero_mask.astype(np.float32), dtype=torch.float32, device=device)
    log_prior_t = torch.tensor(log_pz.astype(np.float32), dtype=torch.float32, device=device)
    log_dt_dz_t = torch.tensor(log_dt_dz.astype(np.float32), dtype=torch.float32, device=device)

    hidden = [int(h) for h in args.hidden.split(",") if h.strip()]
    pi_net = FCNN(1, 1, hidden, nn.ReLU()).to(device)
    flows = []
    for _ in range(int(args.num_transforms)):
        flows.append(ConditionalAffine1D(hidden, s_scale=float(args.s_scale)))
        if bool(getattr(args, "use_tanh_warp", 1)):
            flows.append(ConditionalTanhWarp1D(hidden, c_max=float(getattr(args, "tanh_warp_cmax", 0.85))))
    flow_model = FlowModel(flows).to(device)
    base = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

    opt = torch.optim.AdamW(
        list(pi_net.parameters()) + list(flow_model.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
    )
    tr_t = torch.tensor(tr, dtype=torch.long, device=device)
    te_t = torch.tensor(te, dtype=torch.long, device=device)
    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_t),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    history = {"epoch": [], "train_nll": [], "ul_train": [], "ul_test": []}
    n_epochs = int(getattr(args, "zero_inflated_epochs", args.epochs))
    for epoch in range(n_epochs):
        pi_net.train()
        flow_model.train()
        epoch_loss = 0.0
        n_batches = 0
        for (batch_idx,) in train_dl:
            opt.zero_grad()
            logq = _separated_logq(
                pi_net,
                flow_model,
                base,
                y_t[batch_idx],
                z_s_t[batch_idx],
                zero_t[batch_idx],
                log_dt_dz_t[batch_idx],
                t_sd,
            )
            loss = -logq.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pi_net.parameters()) + list(flow_model.parameters()), args.grad_clip)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        pi_net.eval()
        flow_model.eval()
        with torch.no_grad():
            logq_tr = _separated_logq(
                pi_net, flow_model, base, y_t[tr_t], z_s_t[tr_t], zero_t[tr_t], log_dt_dz_t[tr_t], t_sd
            )
            logq_te = _separated_logq(
                pi_net, flow_model, base, y_t[te_t], z_s_t[te_t], zero_t[te_t], log_dt_dz_t[te_t], t_sd
            )
            ul_train = float(torch.mean(logq_tr - log_prior_t[tr_t]).item())
            ul_test = float(torch.mean(logq_te - log_prior_t[te_t]).item())
        history["epoch"].append(epoch)
        history["train_nll"].append(epoch_loss / max(n_batches, 1))
        history["ul_train"].append(ul_train)
        history["ul_test"].append(ul_test)
    artifacts = {
        "pi_net": pi_net,
        "flow_model": flow_model,
        "base": base,
        "history": history,
        "y_mu": float(y_mu),
        "y_sd": float(y_sd),
        "target_mu": float(t_mu),
        "target_sd": float(t_sd),
        "target_transform": {"kind": "separated_log", "eps0": eps0},
        "prior_model": prior_model,
    }
    if return_training_history:
        return ul_train, ul_test, history, artifacts
    return ul_train, ul_test


def _initial_design_from_args(args, d_space, lo=None, hi=None):
    if getattr(args, "opt_init_L", None) is not None and getattr(args, "opt_init_A", None) is not None:
        return np.array([float(args.opt_init_L), float(args.opt_init_A)], dtype=np.float32)
    init_index = getattr(args, "opt_init_index", None)
    if init_index is not None:
        i = int(init_index)
        if i < 0 or i >= len(d_space):
            raise ValueError(f"--opt_init_index must be in [0, {len(d_space) - 1}], got {i}")
        return np.asarray(d_space[i], dtype=np.float32)
    if lo is not None and hi is not None:
        return 0.5 * (np.asarray(lo, dtype=np.float32) + np.asarray(hi, dtype=np.float32))
    return np.asarray(d_space[len(d_space) // 2], dtype=np.float32)


def _z_preprocess_for_separated_nf(z, eps0):
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    bad = z[:, 0] < -float(eps0)
    if np.any(bad):
        raise ValueError(f"z contains {int(np.sum(bad))} values below -eps0.")
    zero_mask = np.isclose(z[:, 0], 0.0, atol=float(eps0))
    pos_mask = (~zero_mask) & (z[:, 0] > float(eps0))
    if not np.any(pos_mask):
        raise ValueError("separated + log NF requires at least one positive z sample.")
    t_pos = np.log(z[pos_mask, 0]).reshape(-1, 1)
    t_mu = float(np.mean(t_pos))
    t_sd = max(float(np.std(t_pos)), 1e-12)
    z_s = np.zeros_like(z, dtype=np.float32)
    z_s[pos_mask, 0] = ((t_pos[:, 0] - t_mu) / t_sd).astype(np.float32)
    log_dt_dz = np.zeros(z.shape[0], dtype=np.float32)
    log_dt_dz[pos_mask] = (-np.log(z[pos_mask, 0])).astype(np.float32)
    return z_s, zero_mask.astype(np.float32), log_dt_dz, t_mu, t_sd


def _freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def _design_step_sizes_from_args(args, lo, hi):
    default = getattr(args, "opt_design_lr", None)
    if default is None:
        default = 0.05
    step_l = getattr(args, "opt_design_step_L", None)
    step_a = getattr(args, "opt_design_step_A", None)
    if step_l is None:
        step_l = default
    if step_a is None:
        step_a = default
    step = np.array([float(step_l), float(step_a)], dtype=np.float32)
    if np.any(step < 0.0):
        raise ValueError("Design step sizes must be nonnegative.")
    span = np.maximum(np.asarray(hi, dtype=np.float32) - np.asarray(lo, dtype=np.float32), 1e-12)
    return np.minimum(step, span).astype(np.float32)


def optimize_design_with_nf(d_space, obs_fn, pred_fn, forward_model, forward_scaler, args):
    if not bool(args.use_surrogate_forward):
        raise ValueError("--optimize_design requires --use_surrogate_forward 1 so h(theta,d) is differentiable.")

    device = torch.device(args.device)
    lo, hi = _design_bounds_from_args(args)
    d0 = np.minimum(np.maximum(_initial_design_from_args(args, d_space, lo=lo, hi=hi), lo), hi)
    d_param = torch.nn.Parameter(torch.tensor(d0, dtype=torch.float32, device=device))
    lo_t = torch.tensor(lo, dtype=torch.float32, device=device)
    hi_t = torch.tensor(hi, dtype=torch.float32, device=device)
    step_t = torch.tensor(_design_step_sizes_from_args(args, lo, hi), dtype=torch.float32, device=device)

    theta_np = sample_theta_prior_6d(int(args.nout), seed=int(args.seed)).astype(np.float32)
    rng = np.random.default_rng(int(args.seed))
    n_noise_reps = int(max(1, getattr(args, "opt_noise_reps", 8)))
    noise_np = rng.normal(0.0, float(args.sigma_noise), size=(n_noise_reps, theta_np.shape[0], 1)).astype(np.float32)
    z_np = pred_fn(theta_np).reshape(-1, 1)
    eps0 = float(getattr(args, "eps0", 1e-12))
    log_pz_np, _ = _fit_hurdle_log_kde_prior(z_np[:, 0], eps0=eps0)
    z_s_np, zero_np, log_dt_dz_np, _, t_sd = _z_preprocess_for_separated_nf(z_np, eps0)

    theta_t = torch.tensor(theta_np, dtype=torch.float32, device=device)
    noise_t = torch.tensor(noise_np, dtype=torch.float32, device=device)
    z_s_t = torch.tensor(z_s_np, dtype=torch.float32, device=device)
    zero_t = torch.tensor(zero_np, dtype=torch.float32, device=device)
    log_dt_dz_t = torch.tensor(log_dt_dz_np, dtype=torch.float32, device=device)
    log_pz_t = torch.tensor(log_pz_np.astype(np.float32), dtype=torch.float32, device=device)

    records = []
    best = {"UL": -float("inf"), "L": float(d0[0]), "A": float(d0[1]), "outer": -1}
    gradient_method = str(getattr(args, "opt_gradient_method", "score")).lower()
    step_mode = str(getattr(args, "opt_design_step_mode", "gradient")).lower()

    nf_args = SimpleNamespace(**vars(args))
    nf_args.epochs = int(getattr(args, "opt_nf_epochs", args.epochs))
    nf_args.save_design_artifacts = 0

    for outer in range(int(args.opt_design_steps)):
        with torch.no_grad():
            d_param.copy_(torch.clamp(d_param, min=lo_t, max=hi_t))
        d_before = d_param.detach().cpu().numpy().copy()
        design_np = d_before.reshape(1, 2)
        # Inner loop: optimize variational/NF parameters lambda at fixed design xi.
        ul_tr, ul_te, history, artifacts = estimate_nf_lower_bound(
            design_np,
            obs_fn,
            pred_fn,
            nf_args,
            return_training_history=True,
        )

        pi_net = artifacts["pi_net"]
        flow_model = artifacts["flow_model"]
        _freeze_module(pi_net)
        _freeze_module(flow_model)

        # Outer loop: after lambda training finishes, take one projected gradient step in xi.
        if d_param.grad is not None:
            d_param.grad.zero_()
        d_t = d_param
        n_design_batch = min(
            int(getattr(args, "opt_design_batch_size", 0) or args.batch_size),
            int(theta_t.shape[0]),
        )
        n_design_batch = max(1, n_design_batch)
        batch_np = rng.choice(int(theta_t.shape[0]), size=n_design_batch, replace=False)
        batch_idx = torch.tensor(batch_np, dtype=torch.long, device=device)
        theta_b = theta_t[batch_idx]
        z_s_b = z_s_t[batch_idx]
        zero_b = zero_t[batch_idx]
        log_dt_dz_b = log_dt_dz_t[batch_idx]
        log_pz_b = log_pz_t[batch_idx]

        h_b = _torch_obs_surrogate(theta_b, d_t, forward_model, forward_scaler, device)
        y_rep = h_b.unsqueeze(0) + noise_t[:n_noise_reps, batch_idx]
        y_s_all = (y_rep.reshape(-1, 1) - float(artifacts["y_mu"])) / max(float(artifacts["y_sd"]), 1e-12)
        z_s_all = z_s_b.repeat(n_noise_reps, 1)
        zero_all = zero_b.repeat(n_noise_reps)
        log_dt_dz_all = log_dt_dz_b.repeat(n_noise_reps)
        log_pz_all = log_pz_b.repeat(n_noise_reps)

        if gradient_method == "pathwise":
            logq_all = _separated_logq(
                pi_net,
                flow_model,
                artifacts["base"],
                y_s_all,
                z_s_all,
                zero_all,
                log_dt_dz_all,
                float(artifacts["target_sd"]),
            )
            ul_design = torch.mean(logq_all - log_pz_all)
            design_objective = ul_design
            score_objective = None
            score_baseline = None
        elif gradient_method == "score":
            with torch.no_grad():
                logq_all = _separated_logq(
                    pi_net,
                    flow_model,
                    artifacts["base"],
                    y_s_all,
                    z_s_all,
                    zero_all,
                    log_dt_dz_all,
                    float(artifacts["target_sd"]),
                )
                ul_design = torch.mean(logq_all - log_pz_all)
                score_baseline = None
            sigma = max(float(args.sigma_noise), 1e-12)
            y_detached = y_rep.detach()
            resid = y_detached - h_b.unsqueeze(0)
            log_likelihood = -0.5 * (resid / sigma).pow(2) - np.log(sigma * np.sqrt(2.0 * np.pi))
            # MC estimator for grad_xi U_L:
            # E[ log q_lambda(H(theta) | Y) * grad_xi log p(Y | theta, xi) ].
            score_weight = logq_all.detach().reshape(n_noise_reps, n_design_batch, 1)
            score_objective = torch.mean(score_weight * log_likelihood)
            design_objective = score_objective
        else:
            raise ValueError("--opt_gradient_method must be 'pathwise' or 'score'.")

        (-design_objective).backward()
        loss_grad = d_param.grad.detach().clone() if d_param.grad is not None else torch.zeros_like(d_param)
        objective_grad = -loss_grad
        grad_norm = float(objective_grad.norm().cpu().item())
        objective_grad_L = float(objective_grad[0].cpu().item())
        objective_grad_A = float(objective_grad[1].cpu().item())
        if step_mode == "signed":
            delta = step_t * torch.sign(objective_grad)
        elif step_mode == "gradient":
            clipped_grad = objective_grad.clone()
            max_norm = float(args.opt_design_grad_clip)
            norm = torch.linalg.vector_norm(clipped_grad)
            if max_norm > 0.0 and norm > max_norm:
                clipped_grad = clipped_grad * (max_norm / torch.clamp(norm, min=1e-12))
            delta = step_t * clipped_grad
        else:
            raise ValueError("--opt_design_step_mode must be 'signed' or 'gradient'.")
        with torch.no_grad():
            d_param.copy_(torch.clamp(d_param + delta, min=lo_t, max=hi_t))
        d_next = d_param.detach().cpu().numpy()
        actual_delta = d_next - d_before
        ul_val = float(ul_design.detach().cpu().item())
        if ul_val > best["UL"]:
            best = {"UL": ul_val, "L": float(d_before[0]), "A": float(d_before[1]), "outer": int(outer)}
        rec = {
            "outer": int(outer),
            "L": float(d_before[0]),
            "A": float(d_before[1]),
            "L_next": float(d_next[0]),
            "A_next": float(d_next[1]),
            "UL_design_grad": ul_val,
            "design_objective": float(design_objective.detach().cpu().item()),
            "gradient_method": gradient_method,
            "step_mode": step_mode,
            "score_objective": (
                float(score_objective.detach().cpu().item()) if score_objective is not None else np.nan
            ),
            "score_baseline": float(score_baseline.detach().cpu().item()) if score_baseline is not None else np.nan,
            "noise_reps": int(n_noise_reps),
            "design_batch_size": int(n_design_batch),
            "UL_train_fit": float(ul_tr),
            "UL_test_fit": float(ul_te),
            "objective_grad_norm": grad_norm,
            "objective_grad_L": objective_grad_L,
            "objective_grad_A": objective_grad_A,
            "step_L": float(step_t[0].detach().cpu().item()),
            "step_A": float(step_t[1].detach().cpu().item()),
            "delta_L": float(actual_delta[0]),
            "delta_A": float(actual_delta[1]),
            "raw_grad_norm": grad_norm,
            "raw_grad_L": float(loss_grad[0].cpu().item()),
            "raw_grad_A": float(loss_grad[1].cpu().item()),
            "precond_raw_grad_norm": np.nan,
            "precond_grad_L": np.nan,
            "precond_grad_A": np.nan,
            "nf_last_train_nll": float(history["train_nll"][-1]) if history and history.get("train_nll") else np.nan,
        }
        records.append(rec)
        print(
            f"[opt {outer+1}/{int(args.opt_design_steps)}] "
            f"L={rec['L']:.4f}->{rec['L_next']:.4f}, A={rec['A']:.4f}->{rec['A_next']:.4f}, "
            f"UL_grad={rec['UL_design_grad']:.6f}, UL_fit_test={rec['UL_test_fit']:.6f}, "
            f"obj_grad=({rec['objective_grad_L']:.3e}, {rec['objective_grad_A']:.3e}), "
            f"delta=({rec['delta_L']:.3e}, {rec['delta_A']:.3e})"
        )

    out = pd.DataFrame(records)
    rs_tag = _threshold_tag(args)
    rs_suffix = f"_{rs_tag}" if rs_tag else ""
    out_name = f"EIG_LB_z_nf_opt_design_{int(args.nout)}{rs_suffix}.csv"
    out_path = os.path.join(args.output_dir, out_name)
    out.to_csv(out_path, index=False)
    print(
        f"Best continuous design: L={best['L']:.6f}, A={best['A']:.6f}, "
        f"UL={best['UL']:.6f} at outer={best['outer']}"
    )
    print(f"Saved optimization trace: {out_path}")
    return out, best


def main(args):
    os.chdir(_repo_root)
    os.makedirs(args.output_dir, exist_ok=True)
    rs_tag = _threshold_tag(args)
    if bool(getattr(args, "save_design_artifacts", 1)):
        artifacts_dir = os.path.join(
            args.output_dir,
            str(getattr(args, "design_artifacts_dir", "NF_design_artifacts")),
            rs_tag,
        )
        os.makedirs(artifacts_dir, exist_ok=True)
    else:
        artifacts_dir = None
    obs_fn, pred_fn, forward_model, forward_scaler = _prepare_models_and_wrappers(args)
    designs = generate_design_candidates(
        threshold=_roll_threshold(args),
        num_l=3,
        Tp=float(args.Tp),
        Hs=float(args.Hs),
        n_L=args.ngrid,
        n_A=args.ngrid,
        L_Tp_range=(float(args.L_Tp_min), float(args.L_Tp_max)),
        A_Hs_range=(float(args.A_Hs_min), float(args.A_Hs_max)),
    )
    d_space = np.array([[d["L"], d["A"]] for d in designs], dtype=float)

    if bool(getattr(args, "optimize_design", 0)):
        optimize_design_with_nf(d_space, obs_fn, pred_fn, forward_model, forward_scaler, args)
        return

    if args.gridTotal is not None and args.gridRun is not None:
        total = int(args.gridTotal)
        run = int(args.gridRun)
        chunk = len(d_space) // total
        s = run * chunk
        e = (run + 1) * chunk if run < total - 1 else len(d_space)
        d_space = d_space[s:e]
        suffix = f"_run{run}"
    else:
        suffix = ""

    rows = []
    for i, d in enumerate(d_space):
        design_start = time.perf_counter()
        design = d.reshape(1, -1)
        ul_tr, ul_te, history, artifacts = estimate_nf_lower_bound(
            design,
            obs_fn,
            pred_fn,
            args,
            return_training_history=True,
        )
        if artifacts_dir is not None:
            tag = f"NF_model_L{d[0]:.1f}_A{d[1]:.1f}_nout{int(args.nout)}_seed{int(args.seed)}"
            ckpt_path = os.path.join(artifacts_dir, f"{tag}.pt")
            hist_path = os.path.join(artifacts_dir, f"{tag}_history.csv")
            torch.save(
                {
                    "model_type": "separated_log_nf",
                    "pi_net_state_dict": artifacts["pi_net"].state_dict(),
                    "flow_model_state_dict": artifacts["flow_model"].state_dict(),
                    "args": vars(args),
                    "design": design.tolist(),
                    "y_mu": artifacts["y_mu"],
                    "y_sd": artifacts["y_sd"],
                    "target_mu": artifacts["target_mu"],
                    "target_sd": artifacts["target_sd"],
                    "target_transform": artifacts["target_transform"],
                    # backward-compatible aliases
                    "z_mu": artifacts["target_mu"],
                    "z_sd": artifacts["target_sd"],
                },
                ckpt_path,
            )
            if history is not None:
                pd.DataFrame(history).to_csv(hist_path, index=False)
        rows.append([d[0], d[1], ul_tr, ul_te])
        elapsed_seconds = time.perf_counter() - design_start
        print(
            f"[{i+1}/{len(d_space)}] L={d[0]:.3f}, A={d[1]:.3f}, "
            f"UL_train={ul_tr:.6f}, UL_test={ul_te:.6f}, "
            f"time={elapsed_seconds:.2f} s",
            flush=True,
        )

    out = pd.DataFrame(rows, columns=["L", "A", "UL_train", "UL_test"])
    rs_suffix = f"_{rs_tag}" if rs_tag else ""
    out_name = f"EIG_LB_z_{args.nout}{rs_suffix}{suffix}.csv"
    out_path = os.path.join(args.output_dir, out_name)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GO OED lower bound with conditional NF")
    p.add_argument("--threshold", type=float, default=None, help="Roll threshold in radians; defaults to the forward checkpoint filename threshold, or 0.25.")
    p.add_argument("--nout", type=int, default=1000, help="Number of prior samples")
    p.add_argument("--ngrid", type=int, default=10, help="Grid size per dimension")
    p.add_argument("--sigma_noise", type=float, default=0.05, help="Observation noise std")
    p.add_argument("--epochs", type=int, default=300, help="NF training epochs per design")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--hidden", type=str, default="64,64,64", help="Comma-separated FC hidden widths for s,t nets")
    p.add_argument("--s_scale", type=float, default=0.35, help="Bound on |s| in affine: s = s_scale * tanh(...)")
    p.add_argument("--num_transforms", type=int, default=6)
    p.add_argument(
        "--use_tanh_warp",
        type=int,
        default=1,
        help="If 1, interleave conditional monotone tanh warps with affine blocks (more expressive 1D flow).",
    )
    p.add_argument(
        "--tanh_warp_cmax",
        type=float,
        default=0.85,
        help="Max |c(y)| for z' = z + c(y)*tanh(z); must be < 1 for monotonicity.",
    )
    p.add_argument(
        "--eps0",
        type=float,
        default=1e-12,
        help="Tolerance for treating z as the zero hurdle mass.",
    )
    p.add_argument(
        "--save_design_artifacts",
        type=int,
        default=1,
        help="If 1, save per-design checkpoint and training history (NLL + UL history).",
    )
    p.add_argument(
        "--design_artifacts_dir",
        type=str,
        default="NF_design_artifacts",
        help="Subdirectory under --output_dir for per-design model/history files.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output_dir", type=str, default="GO/Data")
    p.add_argument("--nGroupsLA", type=int, default=5000)
    p.add_argument("--use_surrogate_forward", type=int, default=1)
    p.add_argument("--use_surrogate_predict", type=int, default=1)
    p.add_argument("--surrogate_forward_model_path", type=str, default="OEDCOMMON/observe_surrogate_mgdl_rs0p25.pt")
    p.add_argument("--surrogate_forward_scaler_path", type=str, default=None)
    p.add_argument("--surrogate_predict_model_path", type=str, default="OEDCOMMON/predict_surrogate_model_rs0p25.pt")
    p.add_argument("--surrogate_predict_scaler_path", type=str, default=None)
    p.add_argument("--gridRun", type=int, default=None)
    p.add_argument("--gridTotal", type=int, default=None)
    p.add_argument("--optimize_design", type=int, default=0, help="If 1, optimize continuous (L,A) by gradient ascent.")
    p.add_argument("--opt_design_steps", type=int, default=20, help="Outer design-gradient steps.")
    p.add_argument("--opt_nf_epochs", type=int, default=100, help="NF training epochs per outer design step.")
    p.add_argument(
        "--opt_gradient_method",
        type=str,
        default="score",
        choices=["pathwise", "score"],
        help="Design-gradient estimator. 'score' uses the MC likelihood-ratio formula for xi.",
    )
    p.add_argument(
        "--opt_design_step_mode",
        type=str,
        default="gradient",
        choices=["gradient", "signed"],
        help="Use gradient-scaled physical updates by default; 'signed' ignores gradient magnitude.",
    )
    p.add_argument(
        "--opt_design_lr",
        type=float,
        default=5e-2,
        help="Fallback per-coordinate physical learning rate when --opt_design_step_L/A are unset.",
    )
    p.add_argument("--opt_design_step_L", type=float, default=None, help="Physical-coordinate learning rate for L.")
    p.add_argument("--opt_design_step_A", type=float, default=None, help="Physical-coordinate learning rate for A.")
    p.add_argument(
        "--opt_design_grad_clip",
        type=float,
        default=10.0,
        help="Gradient norm clip used only when --opt_design_step_mode gradient.",
    )
    p.add_argument(
        "--opt_design_batch_size",
        type=int,
        default=0,
        help="Mini-batch size for design-gradient steps; 0 reuses --batch_size.",
    )
    p.add_argument("--opt_noise_reps", type=int, default=8, help="Noise replicates per theta for design-gradient averaging.")
    p.add_argument("--opt_init_L", type=float, default=None, help="Initial L for continuous design optimization.")
    p.add_argument("--opt_init_A", type=float, default=None, help="Initial A for continuous design optimization.")
    p.add_argument("--opt_init_index", type=int, default=None, help="Initial grid index if opt_init_L/A are not set.")
    p.add_argument("--Tp", type=float, default=15.0, help="Wave period used to convert L/Tp bounds to L.")
    p.add_argument("--Hs", type=float, default=12.0, help="Wave height used to convert A/Hs bounds to A.")
    p.add_argument("--L_Tp_min", type=float, default=0.1)
    p.add_argument("--L_Tp_max", type=float, default=6.0)
    p.add_argument("--A_Hs_min", type=float, default=0.0)
    p.add_argument("--A_Hs_max", type=float, default=1.5)
    main(p.parse_args())
