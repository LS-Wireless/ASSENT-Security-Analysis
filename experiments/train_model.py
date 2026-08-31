"""Train an ASSENT GNN on CPU, reusing the upstream training loop, saving OUR
checkpoint + thresholds under results/ (never touching upstream).

Usage:
  python experiments/train_model.py --arch nnconv --epochs 100
  python experiments/train_model.py --arch gatv2  --epochs 60
Reads the matching upstream metadata.json for hyperparameters (nhidden, split_seed, ...).
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import paths  # noqa: F401

import numpy as np
import torch

ARCH = {"nnconv": "run_03", "transformer": "run_02", "gatv2": "run_04"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH), default="nnconv")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--nsamps", type=int, default=None, help="override nsamps (for quick tests)")
    ap.add_argument("--data_pkl", default=None, help="load gurobipy-free data bundle (for GPU venv)")
    ap.add_argument("--tag", default=None, help="save under checkpoints_<tag>/ (avoid clobbering)")
    args = ap.parse_args()

    os.chdir(paths.UPSTREAM)  # data_loader uses cwd-relative paths
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from src.learning_v4.assent_data_utils_v4 import data_loader
    from src.learning_v4.assent_data_v4 import GraphDataset
    from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer
    from assent_run_utils_v4 import TrainConfig
    from assent_run_v4 import run_training

    run_id = ARCH[args.arch]
    meta = json.load(open(os.path.join("src/learning_v4/checkpoints", run_id, "metadata.json")))
    cfg_d, conf = meta["TrainConfig"], meta["config"]
    if args.epochs is not None:
        cfg_d["epochs"] = args.epochs
    nsamps = args.nsamps or conf["nsamps"]
    seed = conf["SEED"]; split_seed = conf["split_seed"]

    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if args.data_pkl:
        import pickle
        res = pickle.load(open(args.data_pkl, "rb"))
        print(f"[DATA] loaded gurobipy-free bundle: {args.data_pkl}")
    else:
        res = data_loader(nsamps=nsamps, CONSOLE_RUN=True)
    cfg = TrainConfig(**cfg_d)
    # auto-detect device EXACTLY as the authors' code does (cuda if available else cpu)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] training on: {cfg.device}" + (f" ({torch.cuda.get_device_name(0)})" if cfg.device == "cuda" else ""))
    cfg.rf_cap = res["N_RF"]; cfg.K_tx_milp = res["K_tx_milp"]; cfg.K_rx_milp = res["K_rx_milp"]

    N = nsamps
    idx = np.random.RandomState(split_seed).permutation(N)
    ntr, nva = int(0.8*N), int(0.1*N)
    tr_idx, va_idx, te_idx = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]

    ds = GraphDataset(res["G_comm"], res["S_comm"], res["G_sens"], res["alpha"],
                      res["lambda_cu"], res["lambda_tg"], res["solution"], N_RF=res["N_RF"],
                      device="cpu", K_tx_milp=res["K_tx_milp"], K_rx_milp=res["K_rx_milp"])
    tr = DataLoader(Subset(ds, tr_idx.tolist()), batch_size=cfg.batch_size, shuffle=True)
    va = DataLoader(Subset(ds, va_idx.tolist()), batch_size=cfg.batch_size, shuffle=False)

    if args.arch == "gatv2":
        model = ASSENTGNN_GAT(hidden=conf["nhidden"], layers=conf["nlayers"], drop=conf["drop"], heads=conf["gatv2_heads"])
    elif args.arch == "transformer":
        model = ASSENTGNN_Transformer(hidden=conf["nhidden"], layers=conf["nlayers"], drop=conf["drop"],
                                      heads=conf["transformer_heads"], beta=conf["transformer_beta"])
    else:
        model = ASSENTGNN(hidden=conf["nhidden"], layers=conf["nlayers"], drop=conf["drop"])

    print(f"[TRAIN] arch={args.arch} run_id={run_id} nsamps={nsamps} epochs={cfg.epochs} "
          f"hidden={conf['nhidden']} layers={conf['nlayers']} | train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    t0 = time.time()
    model, summary, history = run_training(model, tr, va, cfg, save_model_state=False)
    dur = time.time() - t0

    ck_root = f"checkpoints_{args.tag}" if args.tag else "checkpoints"
    out_dir = os.path.join(paths.RESULTS, ck_root, args.arch)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
    # make summary JSON-safe
    safe = {}
    for k, v in summary.items():
        if isinstance(v, torch.Tensor): safe[k] = float(v.detach().cpu().item())
        elif isinstance(v, (list, tuple)):
            safe[k] = [float(x.detach().cpu().item()) if isinstance(x, torch.Tensor) else x for x in v]
        else: safe[k] = v
    safe["arch"] = args.arch; safe["run_id"] = run_id; safe["split_seed"] = split_seed
    safe["nsamps"] = nsamps; safe["epochs_run"] = cfg.epochs; safe["train_seconds"] = dur
    safe["test_idx"] = te_idx.tolist()
    json.dump(safe, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    print(f"[DONE] {args.arch} in {dur/60:.1f} min | F1x={summary.get('best_f1x'):.3f} "
          f"tau={summary.get('best_f1tau'):.3f} s={summary.get('best_f1s'):.3f} "
          f"ytx={summary.get('best_f1ytx'):.3f} yrx={summary.get('best_f1yrx'):.3f}")
    print(f"[SAVED] {out_dir}")

if __name__ == "__main__":
    main()
