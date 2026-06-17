# Stage-1 MAE pre-trainer for the MRCNN encoder.
import torch

from model.mae import MaskedAutoencoder, recon_loss


class MAEPretrainer:
    def __init__(self, device, afr_reduced_cnn_size=30, out_len=3000,
                 mask_ratio=0.75, lr=1e-3, weight_decay=1e-4, freq_weight=0.0):
        self.device = device
        self.model = MaskedAutoencoder(
            afr_reduced_cnn_size, out_len, mask_ratio).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        self.freq_weight = freq_weight

    def train_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device)
            self.opt.zero_grad()
            recon, mask = self.model(x)
            loss = recon_loss(recon, x, mask, self.freq_weight)
            loss.backward()
            self.opt.step()
            total += loss.item() * len(x)
            n += len(x)
        return total / max(n, 1)

    def encoder_state_dict(self):
        return {f"encoder.{k}": v
                for k, v in self.model.encoder.state_dict().items()}
