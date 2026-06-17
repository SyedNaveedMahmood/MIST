"""End-to-end smoke test on synthetic npz data (no real dataset needed).
Validates: data loading, MRCNN shapes, prototype bottleneck, MAE recon,
WCO, composite loss, and one training+eval step for A1 and A7."""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import make_config
from data_loader.data_loaders import make_loaders, class_weights
from model.mist_sleep import MISTSleep
from model.mae import MaskedAutoencoder, recon_loss, concentrated_mask
from trainer.trainer import Trainer
from trainer.mae_pretrainer import MAEPretrainer


def make_fake_npz(d, n_subj=4, epochs=60, T=3000):
    files = []
    for s in range(n_subj):
        x = np.random.randn(epochs, T).astype(np.float32)
        y = np.random.randint(0, 5, size=epochs).astype(np.int64)
        f = os.path.join(d, f"SC4{s:02d}E0.npz")
        np.savez(f, x=x, y=y)
        files.append(f)
    return files


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as d:
        files = make_fake_npz(d)
        tr_loader, te_loader, tr_ds, _ = make_loaders(files[:3], files[3:],
                                                      batch_size=16, num_workers=0)
        cw = class_weights(tr_ds)
        print("class weights:", cw.tolist())

        # shape check
        xb, yb = next(iter(tr_loader))
        assert xb.shape[1] == 1 and xb.shape[2] == 3000, xb.shape

        # MRCNN output length must equal d_model (80) for the prototype/TCE path
        m = MISTSleep(use_prototype=True, num_prototypes=16)
        feat = m.mrcnn(xb)
        print("MRCNN feat:", tuple(feat.shape))
        assert feat.shape[1] == 30, feat.shape
        logits, aux = m(xb)
        print("logits:", tuple(logits.shape), "q:", tuple(aux["q"].shape))
        assert logits.shape == (16, 5)

        # MAE
        mae = MaskedAutoencoder(mask_ratio=0.75)
        xm, mask = concentrated_mask(xb)
        assert abs(mask.float().mean().item() - 0.75) < 0.02, mask.float().mean()
        recon, mask = mae(xb)
        print("recon:", tuple(recon.shape), "loss:", recon_loss(recon, xb, mask).item())

        # MAE pretrain one epoch -> load encoder
        pre = MAEPretrainer(device)
        print("[MAE] one-epoch loss:", round(pre.train_epoch(tr_loader), 4))
        miss, unexp = m.load_encoder(pre.encoder_state_dict())
        print("encoder load missing/unexpected:", len(miss), len(unexp))
        assert len(unexp) == 0

        # A7 full: prototype + WCO, one train epoch + eval
        tr = Trainer(m, device, class_weight=cw, use_wco=True,
                     lambda_wco=1.0, lambda_div=0.05, lambda_r=0.2)
        tr.init_prototypes(tr_loader)
        stats = tr.train_epoch(tr_loader)
        print("[A7] train stats:", {k: round(v, 4) for k, v in stats.items()})
        ev = tr.evaluate(te_loader)
        print("[A7] eval:", {k: (round(v, 4) if not isinstance(v, list) else
                                  [round(z, 3) for z in v]) for k, v in ev.items()})

        # A1 baseline (no proto, no wco)
        m1 = MISTSleep(use_prototype=False)
        tr1 = Trainer(m1, device, class_weight=cw, use_wco=False)
        s1 = tr1.train_epoch(tr_loader)
        print("[A1] train stats:", {k: round(v, 4) for k, v in s1.items()})

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
