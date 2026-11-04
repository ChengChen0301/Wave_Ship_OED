"""
Prediction model (QoI) for GO OED — thin wrapper around ``OED_common.predictive_model``.

Callers (e.g. ``mainGO.main``) must pass ``LA_samples``, optional ``scores``, and surrogate
``model`` / ``scaler`` objects loaded there.
"""
import numpy as np
from OEDCOMMON.OED_common import predictive_model


def funcPred_GO(
        theta,
        forward_surrogate_model,
        forward_surrogate_scaler,
        predict_surrogate_model,
        predict_surrogate_scaler,
        LA_samples,
        scores,
        device,
        use_forward_surrogate,
        use_predict_surrogate,
        threshold=0.25,
):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n = theta.shape[0]
    z = np.empty(n, dtype=float)
    for i in range(n):
        value = predictive_model(
            theta[i],
            forward_surrogate_model,
            forward_surrogate_scaler,
            predict_surrogate_model,
            predict_surrogate_scaler,
            LA_samples,
            scores=scores,
            device=device,
            use_forward_surrogate=use_forward_surrogate,
            use_predict_surrogate=use_predict_surrogate,
            threshold=threshold,
        )
        value = np.asarray(value, dtype=float)
        if value.size != 1:
            raise ValueError(
                f"predictive_model must return one QoI per theta; got shape {value.shape} "
                f"at theta index {i}."
            )
        z[i] = value.item()
    return z
