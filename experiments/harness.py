"""Shared inference/decode harness for the security experiments.

Threat-model semantics:
  - The CPU builds the graph from the statistics it RECEIVES (possibly falsified by an
    attacker) and the GNN decodes an allocation from that.
  - The real-world IMPACT (objective, feasibility, who-got-served) is then evaluated on
    the TRUE channel, because the physical network is unchanged by the lie.

So: decode on `seen` inputs, evaluate on `true` inputs.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import paths  # noqa: F401
import numpy as np
import torch

os.chdir(paths.UPSTREAM)
from src.learning_v4.assent_data_utils_v4 import AssentData, data_loader
from src.learning_v4.assent_data_v4 import build_graph
from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer

_ARCH_CLS = {"nnconv": ASSENTGNN, "gatv2": ASSENTGNN_GAT, "transformer": ASSENTGNN_Transformer}
_ARCH_RUN = {"nnconv": "run_03", "gatv2": "run_04", "transformer": "run_02"}


def load_results(nsamps=10000):
    return data_loader(nsamps=nsamps, CONSOLE_RUN=True)


def _ckroot(tag):
    return "checkpoints" if tag in ("paper", "none", "") else f"checkpoints_{tag}"


def load_our_model(arch="nnconv", tag="repro"):
    """Load OUR trained checkpoint + thresholds + test_idx from results/checkpoints_<tag>/.
    The default tag is "repro", the 100-epoch reproduction models every number in the paper
    is computed on. Model hyperparameters (hidden/layers/drop/heads/beta) are read from the
    SAME upstream metadata.json that train_model.py used, so the architecture always matches
    the checkpoint."""
    ckpt_dir = os.path.join(paths.RESULTS, _ckroot(tag), arch)
    summ = json.load(open(os.path.join(ckpt_dir, "summary.json")))
    meta = json.load(open(os.path.join(paths.UPSTREAM, "src", "learning_v4", "checkpoints",
                                       _ARCH_RUN[arch], "metadata.json")))
    conf = meta["config"]
    if arch == "gatv2":
        model = _ARCH_CLS[arch](hidden=conf["nhidden"], layers=conf["nlayers"], drop=conf["drop"],
                                heads=conf["gatv2_heads"])
    elif arch == "transformer":
        model = _ARCH_CLS[arch](hidden=conf["nhidden"], layers=conf["nlayers"], drop=conf["drop"],
                                heads=conf["transformer_heads"], beta=conf["transformer_beta"])
    else:
        model = _ARCH_CLS[arch](hidden=conf["nhidden"], layers=conf["nlayers"], drop=conf["drop"])
    state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    thr = {"x": summ["thr_x"], "tau": summ["thr_tau"], "s": summ["thr_s"],
           "ytx": summ["thr_ytx"], "yrx": summ["thr_yrx"]}
    return model, thr, summ


def make_graph(res, i, Gc=None, Sc=None, Gs=None):
    """Build the HeteroData the CPU would see for sample i, with optional overridden channels."""
    inst = AssentData(
        G_comm=(res["G_comm"][i] if Gc is None else Gc),
        S_comm=(res["S_comm"][i] if Sc is None else Sc),
        G_sens=(res["G_sens"][i] if Gs is None else Gs),
        alpha=res["alpha"][i], lambda_cu=res["lambda_cu"][i], lambda_tg=res["lambda_tg"][i],
        solution=None, N_RF=res["N_RF"], device="cpu",
        K_tx_milp=res["K_tx_milp"], K_rx_milp=res["K_rx_milp"],
    )
    return build_graph(inst)


@torch.no_grad()
def decode(model, data, thr):
    """Decode a single graph to a Solution-like dict {x[A,U],tau[A],s[T],y_tx[A,T],y_rx[A,T]}."""
    out = model(data)
    ei_s  = data[("ap", "serves", "user")].edge_index.cpu().numpy()
    ei_tx = data[("ap", "senses_tx", "target")].edge_index.cpu().numpy()
    ei_rx = data[("ap", "senses_rx", "target")].edge_index.cpu().numpy()
    A = data["ap"].x.shape[0]; U = data["user"].x.shape[0]; T = data["target"].x.shape[0]

    px   = torch.sigmoid(out["x_logit"]).flatten().cpu().numpy()
    ptau = torch.sigmoid(out["tau_logit"]).flatten().cpu().numpy()
    ps   = torch.sigmoid(out["s_logit"]).flatten().cpu().numpy()
    pytx = torch.sigmoid(out["ytx_logit"]).flatten().cpu().numpy()
    pyrx = torch.sigmoid(out["yrx_logit"]).flatten().cpu().numpy()

    x = np.zeros((A, U), np.int8); x[ei_s[0], ei_s[1]] = (px > thr["x"]).astype(np.int8)
    ytx = np.zeros((A, T), np.int8); ytx[ei_tx[0], ei_tx[1]] = (pytx > thr["ytx"]).astype(np.int8)
    yrx = np.zeros((A, T), np.int8); yrx[ei_rx[0], ei_rx[1]] = (pyrx > thr["yrx"]).astype(np.int8)
    tau = (ptau > thr["tau"]).astype(np.int8)
    s = (ps > thr["s"]).astype(np.int8)
    return {"x": x, "tau": tau, "s": s, "y_tx": ytx, "y_rx": yrx}


@torch.no_grad()
def decode_probs(model, data):
    """Return GNN probability MATRICES {px[A,U], ptau[A], ps[T], pytx[A,T], pyrx[A,T]}
    (sigmoid of logits scattered into dense arrays). Used by the projection defense."""
    out = model(data)
    ei_s  = data[("ap", "serves", "user")].edge_index.cpu().numpy()
    ei_tx = data[("ap", "senses_tx", "target")].edge_index.cpu().numpy()
    ei_rx = data[("ap", "senses_rx", "target")].edge_index.cpu().numpy()
    A = data["ap"].x.shape[0]; U = data["user"].x.shape[0]; T = data["target"].x.shape[0]
    px = np.zeros((A, U)); px[ei_s[0], ei_s[1]] = torch.sigmoid(out["x_logit"]).flatten().cpu().numpy()
    pytx = np.zeros((A, T)); pytx[ei_tx[0], ei_tx[1]] = torch.sigmoid(out["ytx_logit"]).flatten().cpu().numpy()
    pyrx = np.zeros((A, T)); pyrx[ei_rx[0], ei_rx[1]] = torch.sigmoid(out["yrx_logit"]).flatten().cpu().numpy()
    ptau = torch.sigmoid(out["tau_logit"]).flatten().cpu().numpy()
    ps = torch.sigmoid(out["s_logit"]).flatten().cpu().numpy()
    return {"px": px, "ptau": ptau, "ps": ps, "pytx": pytx, "pyrx": pyrx}


def sol_from_probs(probs, thr):
    """Threshold probability matrices into a solution dict (consistent with decode)."""
    return {"x": (probs["px"] > thr["x"]).astype(np.int8),
            "tau": (probs["ptau"] > thr["tau"]).astype(np.int8),
            "s": (probs["ps"] > thr["s"]).astype(np.int8),
            "y_tx": (probs["pytx"] > thr["ytx"]).astype(np.int8),
            "y_rx": (probs["pyrx"] > thr["yrx"]).astype(np.int8)}


def params_true(res, i, rho_thresh=0.8):
    """Feasibility params evaluated on the TRUE channel for sample i."""
    T = np.asarray(res["G_sens"][i]).shape[-1]
    return {"N_RF": res["N_RF"], "rho_thresh": rho_thresh, "S_comm": res["S_comm"][i],
            "K_tx": res["K_tx_milp"], "K_rx": res["K_rx_milp"], "C_rx": T}


def obj_params_true(res, i):
    return {"G_comm": res["G_comm"][i], "S_comm": res["S_comm"][i], "G_sens": res["G_sens"][i],
            "lambda_cu": res["lambda_cu"][i], "lambda_tg": res["lambda_tg"][i],
            "alpha": res["alpha"][i], "interf_penalty": res.get("interf_penalty", 0.01)}
