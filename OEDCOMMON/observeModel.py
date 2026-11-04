"""
Observation model for GO OED — uses ``OED_common.forward_model_surrogate`` /
``forward_model_true`` only. Surrogate tensors are provided by the caller (e.g. ``main``).

``use_predict_surrogate`` is accepted for API parity with ``funcPred_GO`` but is ignored:
observations are always the forward map h(d, θ), not z(θ).
"""
import numpy as np
from OEDCOMMON.OED_common import forward_model_surrogate, forward_model_true

def funcObs_GO(
    Prior_Param,
    d,
    use_forward_surrogate,
    forward_surrogate_model,
    forward_surrogate_scaler,
    device: str = "cpu",
    threshold: float = 0.25,
):
    theta = np.asarray(Prior_Param, dtype=float)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n_samples = theta.shape[0]
    L = float(d[0, 0])
    A = float(d[0, 1])
    design = {"L": L, "A": A}

    if use_forward_surrogate:
        y_col = np.array(
            [
                forward_model_surrogate(
                    theta[i],
                    design,
                    forward_surrogate_model,
                    forward_surrogate_scaler,
                    device=device,
                )
                for i in range(n_samples)
            ],
            dtype=float,
        )
    else:
        y_col = np.array(
            [
                float(np.asarray(forward_model_true(theta[i], np.array([L, A], dtype=float), threshold=threshold)).ravel()[0])
                for i in range(n_samples)
            ],
            dtype=float,
        )
    g = y_col.reshape(n_samples, 1)
    return g
