import numpy as np

def grune_required_horizon_and_alpha(*, gamma: float, N: int) -> tuple[int, float]:
    """Compute Grüne's horizon threshold N0(gamma) and alpha_N.

    Parameters
    ----------
    gamma : float
        Constant satisfying the Grüne cost controllability inequality.
    N : int
        MPC horizon.

    Returns
    -------
    N_required : int
        Sufficient horizon length.
    alpha_N : float
        Guaranteed decrease factor in the relaxed DP inequality.

    Calculation Details
    -------------------
    $N_{req}(\gamma) = 2 + \frac{\log(\gamma - 1)}{\log(1 + \frac{1}{\gamma - 1})}$
    where $N_{req}(\gamma)$ is rounded up to the next integer and at least 2.

    $\alpha_N = 1 - \frac{1}{(\gamma / (\gamma - 1))^N - 1}$
    """
    if not np.isfinite(gamma) or gamma <= 0.0:
        return 2, float("nan")

    if gamma <= 1.0 + 1e-12:
        return 2, 1.0

    eps = gamma - 1.0
    denom = np.log1p(1.0 / eps)
    if (not np.isfinite(denom)) or (denom <= 0.0):
        return 2, float("nan")

    N0_real = 2.0 + (np.log(eps) / denom)
    N_required = max(2, int(np.ceil(float(N0_real))))

    ratio = gamma / eps
    log_ratio = np.log(ratio)
    den = np.expm1(float(N-1) * float(log_ratio))
    if (not np.isfinite(den)) or (den <= 0.0):
        return N_required, float("nan")

    alpha_N = float(1.0 - eps / den)
    return N_required, alpha_N