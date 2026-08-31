"""Unit tests for attack invariants (synthetic, fast) + harness decode smoke test.

Run: python attacks/test_attacks.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from attacks.falsify import (gain_inflation_targeted, gain_inflation_steal,
                             correlation_deflation, sensing_inflation)

PASS = "PASS"; FAIL = "FAIL"
ok = True

def check(name, cond):
    global ok
    print(f"  [{PASS if cond else FAIL}] {name}")
    ok &= bool(cond)

# Synthetic small instance
rng = np.random.RandomState(0)
A, U, T = 8, 10, 4
Gc = rng.rand(A, U) * 1e-3 + 1e-4
Sc = rng.rand(A, U, U); Sc = 0.5 * (Sc + Sc.transpose(0, 2, 1))
for a in range(A): np.fill_diagonal(Sc[a], 1.0)
Gs = rng.rand(A, A, T) * 1e-9
a_star = 3

print("=== Attack invariants ===")
# 1) no-op at identity strength
g1, s1, x1 = gain_inflation_steal(Gc, Sc, Gs, a_star, 1.0)
check("gain_steal kappa=1 is exact no-op", np.array_equal(g1, Gc) and np.array_equal(s1, Sc) and np.array_equal(x1, Gs))
g1t, _, _ = gain_inflation_targeted(Gc, Sc, Gs, a_star, 1.0)
check("gain_targeted kappa=1 is exact no-op", np.array_equal(g1t, Gc))
_, s1c, _ = correlation_deflation(Gc, Sc, Gs, a_star, 1.0)
check("corr_deflation factor=1 is exact no-op", np.array_equal(s1c, Sc))
_, _, x1s = sensing_inflation(Gc, Sc, Gs, a_star, 1.0)
check("sensing_inflation kappa=1 is exact no-op", np.array_equal(x1s, Gs))

# 2) only a* slice changes
g2, _, _ = gain_inflation_steal(Gc, Sc, Gs, a_star, 4.0)
other = [a for a in range(A) if a != a_star]
changed_s = np.where(~np.isclose(g2, Gc))
check("gain_steal changes exactly one (a*,u*) entry",
      np.array_equal(g2[other], Gc[other]) and changed_s[0].tolist() == [a_star])
g2t, _, _ = gain_inflation_targeted(Gc, Sc, Gs, a_star, 4.0, u_star=2)
changed = np.where(~np.isclose(g2t, Gc))
check("gain_targeted changes exactly one (a*,u*) entry", changed[0].tolist() == [a_star] and changed[1].tolist() == [2])
_, s2c, _ = correlation_deflation(Gc, Sc, Gs, a_star, 0.0)
check("corr_deflation only a* slab, diagonal preserved",
      np.array_equal(s2c[other], Sc[other]) and np.allclose(np.diag(s2c[a_star]), np.diag(Sc[a_star]))
      and np.allclose(s2c[a_star][~np.eye(U, dtype=bool)], 0.0))
_, _, x2s = sensing_inflation(Gc, Sc, Gs, a_star, 4.0)
# only entries touching a* (tx row OR rx col) change, and they scale by EXACTLY kappa
touch = np.zeros((A, A, T), bool); touch[a_star, :, :] = True; touch[:, a_star, :] = True
check("sensing_inflation leaves non-a* legs unchanged", np.array_equal(x2s[~touch], Gs[~touch]))
check("sensing_inflation scales a*'s legs by exactly kappa",
      np.allclose(x2s[touch], Gs[touch] * 4.0, rtol=1e-9, atol=0.0))
check("sensing self-leg (a*,a*) scaled once = kappa (not kappa^2)",
      np.allclose(x2s[a_star, a_star], Gs[a_star, a_star] * 4.0, rtol=1e-9, atol=0.0))

# 3) monotonicity
vals = [gain_inflation_steal(Gc, Sc, Gs, a_star, k)[0][a_star].sum() for k in [1, 2, 4, 8]]
check("gain_steal monotonic in kappa", all(vals[i] < vals[i+1] for i in range(len(vals)-1)))
vals_t = [gain_inflation_targeted(Gc, Sc, Gs, a_star, k)[0][a_star].sum() for k in [1, 2, 4, 8]]
check("gain_targeted monotonic in kappa", all(vals_t[i] < vals_t[i+1] for i in range(len(vals_t)-1)))

print(f"\n=== INVARIANTS OVERALL: {PASS if ok else FAIL} ===")

# ---- Optional harness smoke (needs a trained checkpoint) ----
ckpt = os.path.join(HERE, "..", "results", "checkpoints_repro", "nnconv", "best_model.pt")
if os.path.exists(ckpt):
    print("\n=== Harness decode smoke (using available nnconv checkpoint) ===")
    try:
        from experiments import harness as H
        res = H.load_results(nsamps=200)
        model, thr, summ = H.load_our_model("nnconv")
        g = H.make_graph(res, 0)
        sol = H.decode(model, g, thr)
        shapes = {k: np.asarray(v).shape for k, v in sol.items()}
        good = sol["x"].shape == (8, 10) and sol["tau"].shape == (8,) and sol["s"].shape == (4,)
        check("harness.decode produces correct shapes", good)
        # attack changes SOMETHING across a few samples (pipeline wired)
        from attacks.falsify import gain_inflation_targeted
        diff = 0
        for i in range(20):
            base = H.decode(model, H.make_graph(res, i), thr)
            Gc2, Sc2, Gs2 = gain_inflation_targeted(res["G_comm"][i], res["S_comm"][i], res["G_sens"][i], 0, 16.0)
            atk = H.decode(model, H.make_graph(res, i, Gc=Gc2, Sc=Sc2, Gs=Gs2), thr)
            diff += int(not np.array_equal(base["x"], atk["x"]))
        check(f"targeted attack changes x on >=1/20 samples (got {diff})", diff >= 1)
    except Exception as e:
        import traceback; traceback.print_exc()
        check("harness smoke ran without error", False)
else:
    print("\n(no checkpoint yet -> skipping harness smoke; invariants are model-free)")

sys.exit(0 if ok else 1)
