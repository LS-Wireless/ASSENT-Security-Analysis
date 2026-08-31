"""Falsified-report attacks by a single compromised AP a*.

Each attack returns NEW (Gc, Sc, Gs) arrays with ONLY a*'s reported slice modified,
modelling an AP that lies to the central server. Invariants (unit-tested):
  - kappa==1 (or factor==1) is an exact no-op (identical arrays).
  - only a*'s slice changes; all other entries are bit-identical.
  - effect is monotonic in the attack strength.
"""
import numpy as np


def _copy(Gc, Sc, Gs):
    return (np.array(Gc, dtype=np.float64, copy=True),
            np.array(Sc, dtype=np.float64, copy=True),
            np.array(Gs, dtype=np.float64, copy=True))


def gain_inflation_targeted(Gc, Sc, Gs, a_star, kappa, u_star=None):
    """Inflate attacker a*'s gain to a SINGLE user u* (default: a*'s own strongest user).
    Non-uniform -> changes the per-AP z-score ranking, so it DOES move GNN features."""
    Gc2, Sc2, Gs2 = _copy(Gc, Sc, Gs)
    if u_star is None:
        u_star = int(np.argmax(Gc2[a_star, :]))
    Gc2[a_star, u_star] = Gc2[a_star, u_star] * kappa
    return Gc2, Sc2, Gs2


def gain_inflation_steal(Gc, Sc, Gs, a_star, kappa, u_star=None):
    """SMARTER grab: inflate a*'s gain to a user it does NOT dominate -- the most 'stealable'
    user (largest gap between the best AP's gain and a*'s gain for that user). Unlike
    gain_targeted (which boosts a*'s already-strongest user, often already served by a*), this
    aims to CAPTURE A NEW user, testing whether the GNN is truly robust or just robust to a weak
    attack. Only a*'s slice changes; kappa==1 is a no-op."""
    Gc2, Sc2, Gs2 = _copy(Gc, Sc, Gs)
    if u_star is None:
        deficit = Gc2.max(axis=0) - Gc2[a_star, :]      # how far a* is below the best server, per user
        deficit[np.argmax(Gc2[a_star, :])] = -np.inf    # exclude a*'s own strongest (already served)
        u_star = int(np.argmax(deficit))
    Gc2[a_star, u_star] = Gc2[a_star, u_star] * kappa
    return Gc2, Sc2, Gs2


def correlation_deflation(Gc, Sc, Gs, a_star, factor):
    """Scale attacker a*'s reported user-user correlations toward 0 (factor in [0,1]) to
    escape the interference-exclusion constraint (7g) and the eq.5 penalty, so the attacker
    can co-serve users it should not. Diagonal is left untouched."""
    Gc2, Sc2, Gs2 = _copy(Gc, Sc, Gs)
    A, U, _ = Sc2.shape
    m = ~np.eye(U, dtype=bool)
    Sc2[a_star][m] = Sc2[a_star][m] * factor
    return Gc2, Sc2, Gs2


def sensing_inflation(Gc, Sc, Gs, a_star, kappa):
    """Inflate attacker a*'s bistatic sensing gains on the legs it controls
    (its Tx row Gs[a*,:,:] and Rx column Gs[:,a*,:]) to hijack target sensing roles.
    Each controlled leg is scaled exactly once by kappa -- including the monostatic
    self-leg (a*,a*), which would otherwise be double-counted to kappa**2."""
    Gc2, Sc2, Gs2 = _copy(Gc, Sc, Gs)
    A = Gs2.shape[0]
    rows = np.arange(A)[:, None, None]
    cols = np.arange(A)[None, :, None]
    touch = (rows == a_star) | (cols == a_star)          # [A,A,1] -> broadcasts over T
    Gs2 = np.where(touch, Gs2 * kappa, Gs2)
    return Gc2, Sc2, Gs2


ATTACKS = {
    "gain_targeted": gain_inflation_targeted,
    "gain_steal": gain_inflation_steal,
    "corr_deflation": correlation_deflation,
    "sensing_inflation": sensing_inflation,
}
