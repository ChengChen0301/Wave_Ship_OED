"""
Compute ship roll trajectories for sampled theta values and wave groups.

This is the trajectory-generation part previously kept in GO/test.py. It saves:
    GO/shipdata/thetas.npy
    GO/shipdata/groups.npy or GO/shipdata/groups_from_grid.npy
    GO/shipdata/scores.npy          if available
    GO/shipdata/true_groups_from_pool/true_groups_i.npy
    GO/shipdata/true_groups_from_grid/true_groups_i.npy

Each true_groups_i.npy contains one roll time series per wave group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from OEDCOMMON.OED_common import get_LA_and_scores, sample_theta_prior_6d
from Ship.ship import compute_ship_roll
from Ship.waves import GroupWave


ROOT = Path("/Users/chengchen/Desktop/Ship_OED_new")
SHIPDATA_DIR = ROOT / "GO" / "shipdata"

N_SAMPLES = 2000
SEED = 1
N_GROUPS_LA = 5000
NUM_PROC = 20
NUM_L = 3
TP = 15
HS = 12

A_HS_RANGE = (0.0, 1.5)
L_TP_RANGE = (0.1, 6.0)
N_A = 50
N_L = 50

POOL_TRAJECTORY_DIR = SHIPDATA_DIR / "true_groups_from_pool"
GRID_TRAJECTORY_DIR = SHIPDATA_DIR / "true_groups_from_grid"


def theta_to_kws_ship(theta: np.ndarray) -> dict:
    alpha1, alpha2, beta1, beta2, epsilon1, epsilon2 = theta
    return dict(
        alpha1=alpha1,
        alpha2=alpha2,
        beta1=beta1,
        beta2=beta2,
        ep1=epsilon1 * np.cos(np.pi / 6),
        ep2=epsilon2 * np.sin(np.pi / 6),
    )


def compute_group_single(x, num_l: int = NUM_L, Tp: float = TP, kws_ship: dict | None = None):
    if kws_ship is None:
        kws_ship = {}
    L, A = x
    groupwave = GroupWave(A, L, 0, Tp)
    solver_result = compute_ship_roll(
        groupwave.generate_wave,
        [-num_l * L, num_l * L],
        **kws_ship,
    )
    return solver_result.y[1]


def create_grid_groups(
    *,
    A_Hs_range: tuple[float, float] = A_HS_RANGE,
    L_Tp_range: tuple[float, float] = L_TP_RANGE,
    n_A: int = N_A,
    n_L: int = N_L,
    Hs: float = HS,
    Tp: float = TP,
) -> np.ndarray:
    L_Tp_values = np.linspace(L_Tp_range[0], L_Tp_range[1], n_L)
    A_Hs_values = np.linspace(A_Hs_range[0], A_Hs_range[1], n_A)
    L_Tp_grid, A_Hs_grid = np.meshgrid(L_Tp_values, A_Hs_values, indexing="ij")
    L_grid = L_Tp_grid * float(Tp)
    A_grid = A_Hs_grid * float(Hs)
    return np.column_stack([L_grid.ravel(), A_grid.ravel()])


def load_or_sample_theta(
    *,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    shipdata_dir: Path = SHIPDATA_DIR,
) -> np.ndarray:
    shipdata_dir = Path(shipdata_dir)
    theta_path = shipdata_dir / "thetas.npy"
    if theta_path.exists():
        theta_prior = np.load(theta_path, allow_pickle=True)
        if theta_prior.shape[0] < n_samples:
            raise ValueError(f"{theta_path} has {theta_prior.shape[0]} theta rows; need {n_samples}")
        return np.asarray(theta_prior[:n_samples], dtype=float)

    theta_prior = sample_theta_prior_6d(n_samples, seed=seed)
    shipdata_dir.mkdir(parents=True, exist_ok=True)
    np.save(theta_path, theta_prior)
    return theta_prior


def compute_trajectories_for_groups(
    theta_prior: np.ndarray,
    groups: np.ndarray,
    *,
    output_dir: Path,
    num_proc: int = NUM_PROC,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(theta_prior.shape[0])
    for i, theta in enumerate(theta_prior, start=1):
        kws = theta_to_kws_ship(theta)
        true_groups = Parallel(n_jobs=num_proc)(
            delayed(compute_group_single)(x, kws_ship=kws) for x in groups
        )
        output_path = output_dir / f"true_groups_{i}.npy"
        np.save(output_path, np.asarray(true_groups, dtype=object))
        print(f"[{i}/{n_samples}] saved {output_path}")


def compute_all_group_trajectories(
    *,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    n_groups_la: int = N_GROUPS_LA,
    num_proc: int = NUM_PROC,
    shipdata_dir: Path = SHIPDATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    shipdata_dir = Path(shipdata_dir)
    shipdata_dir.mkdir(parents=True, exist_ok=True)

    theta_prior = load_or_sample_theta(n_samples=n_samples, seed=seed, shipdata_dir=shipdata_dir)
    la_samples, scores = get_LA_and_scores(n_groups=n_groups_la, seed=seed)

    np.save(shipdata_dir / "thetas.npy", theta_prior)
    np.save(shipdata_dir / "groups.npy", la_samples)
    if scores is not None:
        np.save(shipdata_dir / "scores.npy", scores)

    compute_trajectories_for_groups(
        theta_prior,
        la_samples,
        output_dir=shipdata_dir / "true_groups_from_pool",
        num_proc=num_proc,
    )

    print(f"saved thetas: {shipdata_dir / 'thetas.npy'}")
    print(f"saved groups: {shipdata_dir / 'groups.npy'}")
    if scores is not None:
        print(f"saved scores: {shipdata_dir / 'scores.npy'}")
    print(f"saved true groups in: {shipdata_dir / 'true_groups_from_pool'}")
    return theta_prior, la_samples, scores


def compute_grid_group_trajectories(
    *,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    num_proc: int = NUM_PROC,
    shipdata_dir: Path = SHIPDATA_DIR,
    A_Hs_range: tuple[float, float] = A_HS_RANGE,
    L_Tp_range: tuple[float, float] = L_TP_RANGE,
    n_A: int = N_A,
    n_L: int = N_L,
    Hs: float = HS,
    Tp: float = TP,
) -> tuple[np.ndarray, np.ndarray]:
    shipdata_dir = Path(shipdata_dir)
    shipdata_dir.mkdir(parents=True, exist_ok=True)

    theta_prior = load_or_sample_theta(n_samples=n_samples, seed=seed, shipdata_dir=shipdata_dir)
    groups_from_grid = create_grid_groups(
        A_Hs_range=A_Hs_range,
        L_Tp_range=L_Tp_range,
        n_A=n_A,
        n_L=n_L,
        Hs=Hs,
        Tp=Tp,
    )
    np.save(shipdata_dir / "thetas.npy", theta_prior)
    np.save(shipdata_dir / "groups_from_grid.npy", groups_from_grid)

    compute_trajectories_for_groups(
        theta_prior,
        groups_from_grid,
        output_dir=shipdata_dir / "true_groups_from_grid",
        num_proc=num_proc,
    )

    print(f"saved thetas: {shipdata_dir / 'thetas.npy'}")
    print(f"saved grid groups: {shipdata_dir / 'groups_from_grid.npy'}")
    print(f"grid groups shape: {groups_from_grid.shape}")
    print(f"saved true groups in: {shipdata_dir / 'true_groups_from_grid'}")
    return theta_prior, groups_from_grid


def main() -> None:
    compute_grid_group_trajectories()


if __name__ == "__main__":
    main()
