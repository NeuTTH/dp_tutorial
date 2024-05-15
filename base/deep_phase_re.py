import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pytorch_lightning as pl

# from base.pae_network import *
from base.pae_network_new import *


class deep_phase(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_channels,
        time_range,
        lr,
        batch_size,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.bs = batch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.time_range = time_range
        self.window = self.time_range / 100

        self.tpi = Parameter(
            torch.from_numpy(np.array([2.0 * np.pi], dtype=np.float32)),
            requires_grad=False,
        )
        self.args = Parameter(
            torch.from_numpy(
                np.linspace(
                    -self.window / 2, self.window / 2, self.time_range, dtype=np.float32
                )
            ),
            requires_grad=False,
        )
        self.freqs = Parameter(
            torch.fft.rfftfreq(time_range)[1:] * (time_range) / self.window,
            requires_grad=False,
        )

        ## Network
        self._encoder = Encoder(self.in_channels, self.latent_channels, self.time_range)
        self._decoder = Decoder(
            self.latent_channels, self.out_channels, self.time_range
        )

        self._fl = fft_layer(2, self.time_range, self.freqs)
        self._pl = ps_layer(self.time_range, self.latent_channels, self.tpi)

    def forward(self, x):
        latents = self._encoder(x)
        freq, amp, bias, magnitudes = self._fl(latents)
        shift = self._pl(latents)

        ## Possible to calculate phase here too
        freq = torch.unsqueeze(freq, 2)
        amp = torch.unsqueeze(amp, 2)
        bias = torch.unsqueeze(bias, 2)
        shift = torch.unsqueeze(shift, 2)

        L = amp * torch.sin(self.tpi * (freq * self.args + shift)) + bias

        out = self._decoder(L.float())
        params = [freq, amp, bias, shift]
        return out, latents, L, params, magnitudes

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.lr
        )

        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        seq, _ = batch
        recr, latents, _, params, _ = self(seq)

        rec_loss = F.mse_loss(recr, seq)

        if int(batch_idx) % 500 == 0:
            self.log(
                "train_loss", rec_loss, batch_size=self.bs, on_step=False, on_epoch=True
            )

        sch = self.lr_schedulers()
        sch.step()
        return {"loss": rec_loss}

    def validation_step(self, batch, batch_idx):
        seq, _ = batch
        recr, latents, L, params, _ = self(seq)

        rec_loss = F.mse_loss(recr, seq)

        if int(batch_idx) % 2000 == 0:
            self.log(
                "val_loss", rec_loss, batch_size=self.bs, on_step=False, on_epoch=True
            )

            params_np = [i.detach().cpu().numpy() for i in params]

        return {"loss": rec_loss}

    def predict_step(self, batch, batch_idx):
        seq, tsne_state = batch
        recr, latents, L, params, m = self(seq)
        return (
            seq,
            latents,
            recr,
            L,
            params[0],
            params[1],
            params[2],
            params[3],
            tsne_state,
            m,
        )
