import os
import sys
from typing import Tuple, Dict, Optional, Set

# Directory containing this file (GO/) – used to find surrogate when run from this package
_OED_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Shared random seed for reproducibility across methods
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Bounds for the six uncertain parameters in θ: [alpha1, alpha2, beta1, beta2, epsilon1, epsilon2]
UNCERTAIN_BOUNDS = np.array(
    [
        [0.2, 0.5],      # alpha1
        [0.02, 0.15],    # alpha2
        [0.02, 0.08],    # beta1
        [-0.15, -0.05],  # beta2
        [0.005, 0.02],   # epsilon1
        [0.005, 0.02],   # epsilon2
    ]
)
LB = np.array(UNCERTAIN_BOUNDS[:, 0], dtype=np.float64)
UB = np.array(UNCERTAIN_BOUNDS[:, 1], dtype=np.float64)
THETA_FIXED_SHIP = float(np.pi / 6)
_EXCEEDING_F = None


def funcBoundParam(param, lb, ub):
    """Boundary check for MCMC: True if param is inside [lb, ub]."""
    param = np.asarray(param)
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    return np.all(param >= lb) and np.all(param <= ub)


class SurrogateNN(nn.Module):
    """Stable surrogate network definition for inference from checkpoints."""

    def __init__(
        self,
        input_dim=8,
        hidden_dims=(128, 128, 64),
        dropout=0.1,
        use_batch_norm=False,
        nonnegative_output: bool = False,
    ):
        super().__init__()
        self.nonnegative_output = bool(nonnegative_output)
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        y = self.network(x).squeeze()
        if self.nonnegative_output:
            return F.softplus(y)
        return y


class TwoStepPredictSurrogate(nn.Module):
    """Predictive surrogate for z(theta): zero head plus log-positive-z head."""

    def __init__(
        self,
        input_dim=6,
        hidden_dims=(128, 128, 64),
        dropout=0.05,
    ):
        super().__init__()
        self.zero_net = self._build_mlp(input_dim, hidden_dims, dropout)
        self.positive_net = self._build_mlp(input_dim, hidden_dims, dropout)

    @staticmethod
    def _build_mlp(input_dim, hidden_dims, dropout):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        zero_logit = self.zero_net(x).squeeze(-1)
        log_pos_z_s = self.positive_net(x).squeeze(-1)
        return zero_logit, log_pos_z_s


class ShallowGradeNet(nn.Module):
    """One MGDL grade: a shallow scalar correction network."""

    def __init__(
        self,
        in_dim=8,
        hidden_dims=(128, 128, 64),
        dropout=0.05,
        use_batch_norm=True,
    ):
        super().__init__()
        layers = []
        prev_dim = int(in_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ObserveSurrogateMGDL(nn.Module):
    """MGDL observe surrogate: sum of residual grade networks."""

    def __init__(
        self,
        base_dim=8,
        n_grades=5,
        grade_hidden_dims=(128, 128, 64),
        dropout=0.05,
        use_batch_norm=True,
    ):
        super().__init__()
        self.base_dim = int(base_dim)
        self.n_grades = int(n_grades)
        self.grade_hidden_dims = tuple(int(h) for h in grade_hidden_dims)
        self.dropout = float(dropout)
        self.use_bn = bool(use_batch_norm)
        self.grades = nn.ModuleList(
            [
                ShallowGradeNet(
                    self.base_dim + k,
                    hidden_dims=self.grade_hidden_dims,
                    dropout=self.dropout,
                    use_batch_norm=self.use_bn,
                )
                for k in range(self.n_grades)
            ]
        )

    def forward(self, x):
        feats = [x]
        total = x.new_zeros(x.size(0))
        for net in self.grades:
            inp = torch.cat(feats, dim=1)
            out = net(inp)
            total = total + out
            feats.append(out.unsqueeze(1))
        return total


def _load_scaler_payload(scaler_path: str) -> StandardScaler:
    scaler_data = np.load(scaler_path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["scaler_mean"]
    scaler.scale_ = scaler_data["scaler_scale"]
    scaler.var_ = scaler_data["scaler_var"]
    if "y_mean" in scaler_data and "y_scale" in scaler_data:
        scaler.y_mean_ = scaler_data["y_mean"]
        scaler.y_scale_ = scaler_data["y_scale"]
    return scaler


def _load_surrogate_checkpoint(model_path: str, device: str = "cpu"):
    try:
        ck = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(model_path, map_location=device)
    cfg = ck["model_config"]
    sd = ck.get("model_state_dict")
    if sd is None:
        sd = ck.get("state_dict")
    if sd is None:
        raise KeyError("Checkpoint missing model_state_dict/state_dict.")
    return cfg, sd


def _checkpoint_scaler_payload(ck: dict) -> StandardScaler:
    scaler_data = ck.get("input_scaler")
    if scaler_data is None:
        scaler_data = ck.get("theta_scaler")
    if scaler_data is None:
        raise KeyError("Checkpoint missing input_scaler/theta_scaler; provide a separate scaler_path.")

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(scaler_data["mean"], dtype=float)
    scaler.scale_ = np.asarray(scaler_data["scale"], dtype=float)
    scaler.var_ = np.asarray(scaler_data.get("var", scaler.scale_ ** 2), dtype=float)

    output_scaler = ck.get("output_scaler")
    if output_scaler is not None:
        scaler.y_mean_ = np.asarray(output_scaler["mean"], dtype=float)
        scaler.y_scale_ = np.asarray(output_scaler["scale"], dtype=float)
    positive_target = ck.get("positive_target")
    if positive_target is not None:
        scaler.y_mean_ = np.asarray(positive_target["mean"], dtype=float)
        scaler.y_scale_ = np.asarray(positive_target["scale"], dtype=float)
        scaler.target_transform_ = str(positive_target.get("transform", "log"))
    if "r_s" in ck:
        scaler.r_s_ = float(ck["r_s"])
    return scaler


def _build_model_from_checkpoint(cfg: dict, state_dict: dict, device: str = "cpu", model_type: Optional[str] = None):
    if model_type == "observe_surrogate_h_mgdl" or bool(cfg.get("mgdl", False)) or any(
        k.startswith("grades.") for k in state_dict
    ):
        model = ObserveSurrogateMGDL(
            base_dim=cfg["base_dim"],
            n_grades=cfg["n_grades"],
            grade_hidden_dims=tuple(cfg["grade_hidden_dims"]),
            dropout=cfg.get("dropout", 0.05),
            use_batch_norm=cfg.get("use_batch_norm", True),
        )
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        return model

    if model_type == "two_step_predict_surrogate" or any(k.startswith("zero_net.") for k in state_dict):
        model = TwoStepPredictSurrogate(
            input_dim=cfg["input_dim"],
            hidden_dims=tuple(cfg["hidden_dims"]),
            dropout=cfg["dropout"],
        )
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        return model

    # Try both BN layouts because some historical checkpoints saved wrong flags.
    last_err = None
    nn_out = bool(cfg.get("nonnegative_output", False))
    for use_bn_try in (True, False):
        try:
            model = SurrogateNN(
                input_dim=cfg["input_dim"],
                hidden_dims=tuple(cfg["hidden_dims"]),
                dropout=cfg["dropout"],
                use_batch_norm=use_bn_try,
                nonnegative_output=nn_out,
            )
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)
            model.eval()
            return model
        except RuntimeError as e:
            last_err = e
    raise RuntimeError(f"Could not build SurrogateNN from checkpoint. Last error: {last_err}") from last_err


def get_surrogate_model_and_scaler(
        model_path,
        scaler_path,
        device: str = "cpu",
):
    try:
        ck = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(model_path, map_location=device)
    cfg = ck["model_config"]
    sd = ck.get("model_state_dict")
    if sd is None:
        sd = ck.get("state_dict")
    if sd is None:
        raise KeyError("Checkpoint missing model_state_dict/state_dict.")

    if scaler_path is None:
        scaler = _checkpoint_scaler_payload(ck)
    else:
        scaler = _load_scaler_payload(scaler_path)
    model = _build_model_from_checkpoint(cfg, sd, device=device, model_type=ck.get("model_type"))
    return model, scaler


def surrogate_map_generic(model, scaler, X, device: str = "cpu"):
    """Predict helper with optional inverse y scaling from scaler payload."""
    model.eval()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_s = scaler.transform(X)
    with torch.no_grad():
        out = model(torch.FloatTensor(X_s).to(device))
    if isinstance(out, tuple):
        zero_logit, log_pos_z_s = out
        p_zero = torch.sigmoid(zero_logit).cpu().numpy()
        log_pos_z_s = log_pos_z_s.cpu().numpy()
        ym = np.asarray(getattr(scaler, "y_mean_"), dtype=float).ravel()[0]
        ys = np.asarray(getattr(scaler, "y_scale_"), dtype=float).ravel()[0]
        z_pos = np.exp(log_pos_z_s * ys + ym)
        if np.isclose(float(getattr(scaler, "r_s_", np.nan)), 0.1):
            return z_pos
        return np.where(p_zero >= 0.5, 0.0, z_pos)
    out_z = out.cpu().numpy()
    out_z = np.atleast_1d(out_z)
    ym = getattr(scaler, "y_mean_", None)
    ys = getattr(scaler, "y_scale_", None)
    if ym is not None and ys is not None:
        ym = np.asarray(ym, dtype=float).ravel()
        ys = np.asarray(ys, dtype=float).ravel()
        y = out_z * ys + ym
        if str(getattr(scaler, "target_transform_", "")).lower() == "log":
            return np.exp(y)
        return y
    return out_z


def sample_theta_prior_6d(n_samples: int = 1000, seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Sample parameters from the uniform prior:
        [alpha1, alpha2, beta1, beta2, epsilon1, epsilon2].
    """
    rng = np.random.default_rng(int(seed))
    return rng.uniform(
        low=UNCERTAIN_BOUNDS[:, 0],
        high=UNCERTAIN_BOUNDS[:, 1],
        size=(n_samples, 6),
    )


sample_theta_prior = sample_theta_prior_6d


def log_prior(theta: np.ndarray) -> float:
    """
    Log of the uniform prior over the six uncertain parameters:
        [alpha1, alpha2, beta1, beta2, epsilon1, epsilon2].
    """
    theta = np.asarray(theta)

    in_bounds = np.logical_and(
        theta >= UNCERTAIN_BOUNDS[:, 0],
        theta <= UNCERTAIN_BOUNDS[:, 1],
    )
    if not np.all(in_bounds):
        return -np.inf

    widths = UNCERTAIN_BOUNDS[:, 1] - UNCERTAIN_BOUNDS[:, 0]
    log_norm = -np.sum(np.log(widths))
    return float(log_norm)


def log_likelihood(
        theta: np.ndarray,
        y: float,
        design: Dict[str, float],
        sigma_noise: float = 0.05,
        use_forward_surrogate=True,
        threshold=0.25,
) -> float:
    """
    Scalar Gaussian log-likelihood log p(y | θ, d) with surrogate mean.
    """
    if use_forward_surrogate:
        mu = forward_model_surrogate(theta, design)
    else:
        mu = forward_model_true(theta, design, threshold)
    resid = y - mu
    return float(
        -0.5 * (resid / sigma_noise) ** 2
        - 0.5 * np.log(2.0 * np.pi * sigma_noise**2)
    )


def log_posterior(
        theta: np.ndarray,
        y: float,
        design: Dict[str, float],
        sigma_noise: float = 0.05,
        use_forward_surrogate=True,
        threshold=0.25,
) -> float:
    """
    Unnormalized log-posterior log p(θ | y, d) ∝ log p(y | θ, d) + log p(θ).
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, y, design, sigma_noise, use_forward_surrogate, threshold)
    return lp + ll


def forward_model_surrogate(
    theta,
    design,
    model,
    scaler,
    device: str = "cpu",
):
    """
    Forward surrogate model: maps (theta, design) -> scalar exceeding metric.
    """
    L = design["L"]
    A = design["A"]

    # Input: [L, A, alpha1, alpha2, beta1, beta2, epsilon1, epsilon2]
    # Accept 6D (uncertain params only) or 7D (with fixed angle at index 4)
    theta = np.asarray(theta).ravel()
    a1, a2, b1, b2, e1, e2 = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
    X = np.array([[L, A, a1, a2, b1, b2, e1, e2]])

    y_pred = surrogate_map_generic(model, scaler, X, device=device)
    return float(np.asarray(y_pred).ravel()[0])


def forward_model_true(
        theta,
        design,
        threshold: float = 0.25,
):
    theta_6d = _theta_ship_6d(theta)
    kws_ship = _kws_ship_from_theta_6d(theta_6d)
    from Ship.exceeding import f as exceeding_f
    y_pred = exceeding_f(
        design,
        threshold,
        kws_ship=kws_ship,
    )
    y_pred = np.atleast_1d(np.asarray(y_pred, dtype=float).ravel())
    return y_pred


_REPO_ROOT = os.path.abspath(os.path.join(_OED_COMMON_DIR, ".."))


def _try_load_la_scores_pair(samples_path: str, scores_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        samples = np.load(samples_path, allow_pickle=True)
        scores = np.load(scores_path, allow_pickle=True)
        scores = np.atleast_1d(np.asarray(scores).ravel())
        if samples.ndim == 1:
            LA_samples = np.array([[s[0], s[1]] for s in samples], dtype=float)
        else:
            LA_samples = np.asarray(samples[:, :2], dtype=float)
        if LA_samples.shape[0] != scores.shape[0]:
            return None, None
        return LA_samples, scores
    except Exception:
        return None, None


def load_LA_and_scores_from_wavedata(
    samples_path: Optional[str] = None,
    scores_path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load fixed detected groups (L, A) and scores from wavedata.

    If ``samples_path`` and ``scores_path`` are both given, only that pair is tried.

    Otherwise tries, in order (first successful load wins):

    - ``wavedata/samples.npy`` relative to cwd
    - ``<repo>/OEDCOMMON/wavedata/samples.npy``
    - ``<repo>/Ship/wavedata/samples.npy``
    - ``<repo>/wavedata/samples.npy``

    (with matching ``scores.npy`` next to each ``samples.npy``).

    Returns (None, None) if no pair loads or row counts mismatch.
    """
    if samples_path is not None and scores_path is not None:
        return _try_load_la_scores_pair(samples_path, scores_path)

    pairs: list[Tuple[str, str]] = [
        ("wavedata/samples.npy", "wavedata/scores.npy"),
        (
            os.path.join(_REPO_ROOT, "OEDCOMMON", "wavedata", "samples.npy"),
            os.path.join(_REPO_ROOT, "OEDCOMMON", "wavedata", "scores.npy"),
        ),
        (
            os.path.join(_REPO_ROOT, "Ship", "wavedata", "samples.npy"),
            os.path.join(_REPO_ROOT, "Ship", "wavedata", "scores.npy"),
        ),
        (
            os.path.join(_REPO_ROOT, "wavedata", "samples.npy"),
            os.path.join(_REPO_ROOT, "wavedata", "scores.npy"),
        ),
    ]
    seen: Set[Tuple[str, str]] = set()
    for sp, sc in pairs:
        key = (os.path.abspath(sp), os.path.abspath(sc))
        if key in seen:
            continue
        seen.add(key)
        la, s = _try_load_la_scores_pair(sp, sc)
        if la is not None:
            return la, s
    return None, None


def get_LA_and_scores(
    n_groups: Optional[int] = None,
    seed: int = RANDOM_SEED,
    replace: bool = False,
    samples_path: Optional[str] = None,
    scores_path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    LA_samples, scores = load_LA_and_scores_from_wavedata(samples_path, scores_path)

    if LA_samples is None:
        return None, None
    LA = np.asarray(LA_samples, dtype=float)
    if LA.ndim != 2 or LA.shape[0] == 0:
        return None, None
    scores_arr = None if scores is None else np.asarray(scores, dtype=float).ravel()

    if n_groups is None:
        return LA, scores_arr

    n_groups = int(n_groups)
    if n_groups <= 0:
        raise ValueError("n_groups must be a positive integer")

    n_pool = LA.shape[0]
    if (not replace) and (n_groups >= n_pool):
        return LA, scores_arr

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pool, size=n_groups, replace=bool(replace))
    LA_sub = LA[idx]
    scores_sub = None if scores_arr is None else scores_arr[idx]
    return LA_sub, scores_sub


def generate_design_candidates(
    threshold: float = 0.25,
    num_l: int = 3,
    Tp: float = 15,
    Hs: float = 12,
    n_L: int = 30,
    n_A: int = 30,
    L_Tp_range: Tuple[float, float] = (0.0, 6.0),
    A_Hs_range: Tuple[float, float] = (0.0, 1.5),
):
    """
    Generate candidate (L, A) designs on a normalized grid.
    """
    L_Tp_values = np.linspace(L_Tp_range[0], L_Tp_range[1], n_L)
    A_Hs_values = np.linspace(A_Hs_range[0], A_Hs_range[1], n_A)

    designs = []
    for L_Tp in L_Tp_values:
        for A_Hs in A_Hs_values:
            L = L_Tp * Tp
            A = A_Hs * Hs
            designs.append(
                {
                    "L": L,
                    "A": A,
                    "threshold": threshold,
                    "num_l": num_l,
                    "Tp": Tp,
                }
            )
    return designs


def get_bandwidth_range(X, method: str = "all", n_points: int = 20):
    """
    Generate a reasonable bandwidth range for KDE (1D).
    """
    X = np.asarray(X)
    if X.ndim > 1:
        X = X.ravel()

    n = len(X)
    if n <= 1:
        return np.linspace(0.1, 1.0, n_points), {}

    std = np.std(X)
    iqr = np.percentile(X, 75) - np.percentile(X, 25)

    h_scott = std * (n ** (-1 / 5))
    h_silverman = 0.9 * std * (n ** (-1 / 5))
    h_iqr = 0.9 * min(std, iqr / 1.34) * (n ** (-1 / 5))

    gamma = 0.92 * std * (n ** (-1 / 13))
    try:
        pilot_kde = gaussian_kde(X.reshape(1, -1), bw_method=gamma)
        x_grid = np.linspace(np.min(X), np.max(X), 100)
        pilot_density = pilot_kde.evaluate(x_grid.reshape(1, -1))
        h_dpi = (1 / (n * np.sum(np.gradient(np.gradient(pilot_density)) ** 2))) ** (1 / 5)
    except Exception:
        h_dpi = h_scott

    reference_values = {
        "scott": h_scott,
        "silverman": h_silverman,
        "robust_silverman": h_iqr,
        "dpi": h_dpi,
    }

    if method == "scott":
        h_min = h_scott / 2
        h_max = h_scott * 2
    elif method == "silverman":
        h_min = h_silverman / 2
        h_max = h_silverman * 2
    elif method == "dpi":
        h_min = h_dpi / 2
        h_max = h_dpi * 2
    else:
        h_min = min(h_scott, h_silverman, h_iqr, h_dpi) / 2
        h_max = max(h_scott, h_silverman, h_iqr, h_dpi) * 2

    h_range = np.linspace(h_min, h_max, n_points)
    return h_range, reference_values


def kde_cv_score(h, X_train, X_val):
    """
    Calculate the log-likelihood score for KDE with bandwidth h.
    Higher scores are better (we want to maximize log-likelihood).
    """
    try:
        # Fit KDE on training data with specified bandwidth
        kde = gaussian_kde(X_train, bw_method=h)
        densities = kde.evaluate(X_val)
        eps = np.finfo(float).eps
        densities = np.maximum(densities, eps)

        # Calculate log-likelihood on validation set
        log_likelihood = np.sum(np.log(densities))

        # Return log-likelihood (we want to maximize this)
        return log_likelihood
    except:
        # Return large negative value if KDE fails
        return -np.inf


def select_optimal_bandwidth(X, h_range=None, n_folds: int = 5, random_state: int = 42):
    """
    K-fold CV to select KDE bandwidth (1D data).
    """
    X = np.array(X).reshape(-1, 1)
    if h_range is None:
        scott_h = X.std() * (len(X) ** (-1 / 5))
        h_range = np.linspace(scott_h / 2, scott_h * 2, 20)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_scores = {h: [] for h in h_range}

    for train_idx, val_idx in kf.split(X):
        X_train = X[train_idx].ravel()
        X_val = X[val_idx].ravel()
        for h in h_range:
            score = kde_cv_score(h, X_train, X_val)
            cv_scores[h].append(score)

    mean_scores = {h: np.mean(scores) for h, scores in cv_scores.items()}
    valid_scores = {h: score for h, score in mean_scores.items() if np.isfinite(score)}
    if not valid_scores:
        raise ValueError("No valid bandwidths found. Try a different range.")

    optimal_h = max(valid_scores.items(), key=lambda x: x[1])[0]
    optimal_score = valid_scores[optimal_h]
    return optimal_h, optimal_score, mean_scores


def forward_surrogate_outputs_batch(
        theta,
        LA_samples,
        model=None,
        scaler=None,
        device: str = "cpu",
):
    """
    Return surrogate output (exceeding metric) for each (L, A) in LA_samples.
    """
    if model is None or scaler is None:
        model, scaler = get_surrogate_model_and_scaler(device=device)

    LA_samples = np.asarray(LA_samples)
    if LA_samples.ndim != 2 or LA_samples.shape[1] != 2:
        raise ValueError("LA_samples must be an array of shape (n_samples, 2)")

    n_samples = LA_samples.shape[0]
    X_batch = np.zeros((n_samples, 8))
    X_batch[:, 0] = LA_samples[:, 0]
    X_batch[:, 1] = LA_samples[:, 1]
    theta = np.asarray(theta).ravel()
    if theta.shape[0] == 6:
        X_batch[:, 2:6] = theta[:4]
        X_batch[:, 6:8] = theta[4:6]
    else:
        X_batch[:, 2:6] = theta[:4]
        X_batch[:, 6:8] = theta[5:7]

    y_pred = surrogate_map_generic(model, scaler, X_batch, device=device)
    return np.atleast_1d(np.asarray(y_pred).ravel())


def _theta_ship_6d(theta: np.ndarray) -> np.ndarray:
    """Six uncertain ship parameters; supports 6D θ or 7D with a fixed slot at index 4."""
    theta = np.asarray(theta).ravel()
    if theta.shape[0] == 6:
        return theta
    if theta.shape[0] == 7:
        return np.array([theta[0], theta[1], theta[2], theta[3], theta[5], theta[6]])
    raise ValueError(f"theta must have length 6 or 7, got {theta.shape[0]}")


def _kws_ship_from_theta_6d(theta_6d: np.ndarray) -> dict:
    alpha1, alpha2, beta1, beta2, epsilon1, epsilon2 = np.asarray(theta_6d).ravel()[:6]
    ep1 = epsilon1 * np.cos(THETA_FIXED_SHIP)
    ep2 = epsilon2 * np.sin(THETA_FIXED_SHIP)
    return dict(
        alpha1=alpha1,
        alpha2=alpha2,
        beta1=beta1,
        beta2=beta2,
        ep1=ep1,
        ep2=ep2,
    )


def predictive_model(
        theta,
        forward_surrogate_model,
        forward_surrogate_scaler,
        predict_surrogate_model,
        predict_surrogate_scaler,
        LA_samples,
        scores=None,
        device: str = "cpu",
        Tp=15,
        base_scale=256,
        num_l=3,
        n_pool=289343,
        D_thr=0.25, # threshold for scores
        use_forward_surrogate=True,
        use_predict_surrogate=True,
        threshold=0.25,
):
    """
    Predictive model: exceeding metric plus recovery step.
    """
    if use_predict_surrogate:
        z = surrogate_map_generic(predict_surrogate_model, predict_surrogate_scaler, theta, device=device)

    else:  # will use the LA integral
        LA_samples = np.asarray(LA_samples)
        n_samples = LA_samples.shape[0]
        if LA_samples.ndim != 2 or LA_samples.shape[1] != 2:
            raise ValueError("LA_samples must be an array of shape (n_samples, 2)")

        if use_forward_surrogate:  # Use surrogate forward model to get the exceeding metric then recovery
            y_pred = forward_surrogate_outputs_batch(theta, LA_samples, forward_surrogate_model, forward_surrogate_scaler, device=device)
        else:  # Use the true ODEsolver forward model to get the exceeding metric then recovery
            y_pred = forward_model_true(theta, LA_samples, threshold)

        # Recovery: surrogate auxiliary in [-1, 1] -> nonnegative; same clip for true f
        recovered = np.maximum(y_pred, 0.0)

        if scores is not None:
            # P^a_group following notebook: score filter then fraction that exceed
            scores = np.atleast_1d(np.asarray(scores).ravel())
            if scores.shape[0] != LA_samples.shape[0]:
                raise ValueError("scores must have the same length as LA_samples")
            high = scores >= (1 - D_thr)
            n_high = np.sum(high)
            if n_high == 0:
                return 0.0
            return np.sum(recovered * LA_samples[:, 0] * 2 * num_l * high) * (n_pool / n_samples) / (Tp * base_scale * 9499)

        z = np.sum(recovered * LA_samples[:, 0] * 2 * num_l) * (n_pool / n_samples) / (Tp * base_scale * 9499)
    return z


def plot_bandwidth_range(X, method='all'):
    """
    Visualize the bandwidth range and different rule-of-thumb estimates
    """
    # Ensure X is properly shaped
    X = np.asarray(X)
    if X.ndim > 1:
        X = X.ravel()

    h_range, reference_values = get_bandwidth_range(X, method=method)

    # Create KDE plots for different bandwidths
    x_grid = np.linspace(np.min(X) - np.std(X), np.max(X) + np.std(X), 200)

    plt.figure(figsize=(12, 6))

    # Plot KDE with different bandwidths
    for h in [h_range[0], np.median(h_range), h_range[-1]]:
        kde = gaussian_kde(X.reshape(1, -1), bw_method=h)
        plt.plot(x_grid, kde.evaluate(x_grid.reshape(1, -1)),
                 label=f'h = {h:.4f}', alpha=0.5)

    # Plot reference bandwidths
    colors = ['r', 'g', 'b', 'purple']
    for (name, h), color in zip(reference_values.items(), colors):
        kde = gaussian_kde(X.reshape(1, -1), bw_method=h)
        plt.plot(x_grid, kde.evaluate(x_grid.reshape(1, -1)),
                 color=color, linestyle='--',
                 label=f'{name}: h = {h:.4f}')

    # Plot data points
    plt.hist(X, bins=30, density=True, alpha=0.3, color='gray')

    plt.title('KDE with Different Bandwidths')
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    return plt
