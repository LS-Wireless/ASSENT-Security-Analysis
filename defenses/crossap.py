"""Input-side defense for the interference (correlation-deflation) attack: a CROSS-AP CONSISTENCY
check the server CAN actually run.

Idea: the attacker a* deflates its own reported interference S[a*,u,u'] to sneak clashing users
together. But every AP reports interference for the same user pairs, and a user pair that truly
clashes tends to look correlated to MANY APs. So the server compares each AP's reported S against
the consensus (median over the OTHER APs) per user pair, flags the AP whose row sits systematically
below the consensus (the deflator), and lifts its flagged entries up toward the consensus. This uses
only reported data from the honest APs -- no raw CSI, no oracle.

sanitize_scomm(Sc, rho_thresh=0.8) -> (Sc_clean, flagged_ap):
  detects the single most-deflated AP and repairs its below-consensus off-diagonal entries.
"""
import numpy as np


def sanitize_scomm(Sc, z_flag=1.0, lift=0.75):
    """Sc: [A,U,U] reported user-user interference. Returns (cleaned Sc, flagged AP index or -1).
    z_flag: how many (robust) sigmas below consensus counts as suspicious.
    lift: fraction of the way to lift a flagged entry from its value toward the consensus."""
    Sc = np.array(Sc, dtype=np.float64, copy=True)
    A, U, _ = Sc.shape
    if A < 3:
        return Sc, -1
    iu, ju = np.triu_indices(U, k=1)                       # off-diagonal pairs
    pairs = Sc[:, iu, ju]                                  # [A, P] each AP's reported value per pair
    # consensus per pair = median over APs; spread = MAD (robust)
    med = np.median(pairs, axis=0)                         # [P]
    mad = np.median(np.abs(pairs - med), axis=0) + 1e-6    # [P]
    below = (med - pairs) / (1.4826 * mad)                 # [A,P] how many robust sigmas BELOW consensus
    # deflation score per AP = mean amount it sits below consensus on pairs where consensus is non-trivial
    mask = med > 0.3                                       # only judge pairs that are correlated in consensus
    score = (np.clip(below, 0, None) * mask).mean(axis=1)  # [A]
    a_flag = int(np.argmax(score))
    if score[a_flag] < z_flag:                             # nobody clearly deflating
        return Sc, -1
    # lift the flagged AP's below-consensus entries toward the consensus
    row = pairs[a_flag].copy()
    susp = (below[a_flag] > z_flag) & mask
    row[susp] = row[susp] + lift * (med[susp] - row[susp])
    Sc[a_flag, iu, ju] = row
    Sc[a_flag, ju, iu] = row                               # keep symmetric
    return Sc, a_flag
