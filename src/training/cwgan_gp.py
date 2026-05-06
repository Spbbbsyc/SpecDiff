from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from evaluation.plots import plot_gan_losses, plot_samples
from models.cwgan_gp import Discriminator, Generator
from utils import get_device

matplotlib.use("Agg")

DEFAULTS = dict(
    data_path="processed_spectra_split_70_15_15.npz",
    output_dir="runs/gan",
    latent_dim=128,
    n_classes=None,
    spectrum_len=1024,
    batch_size=128,
    epochs=500,
    lr_g=1e-4,
    lr_d=1e-4,
    n_critic=5,
    gp_lambda=10.0,
    ac_lambda=1.0,
    label_emb_dim=128,
    beta1=0.0,
    beta2=0.9,
    save_every=50,
    plot_every=25,
    seed=42,
)


def gradient_penalty(discriminator: Discriminator, real: torch.Tensor, fake: torch.Tensor, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interpolated, labels)
    grads = torch.autograd.grad(outputs=d_interp, inputs=interpolated, grad_outputs=torch.ones_like(d_interp), create_graph=True, retain_graph=True)[0]
    grads = grads.view(batch_size, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def load_data(data_path: str, batch_size: int) -> tuple[DataLoader, np.ndarray, np.ndarray, int, int]:
    data = np.load(data_path, allow_pickle=True)
    if "train_spectra" in data:
        spectra = data["train_spectra"]
        labels = data["train_labels"]
    else:
        spectra = data["spectra"]
        labels = data["labels"]
    dataset = TensorDataset(torch.tensor(spectra, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True), data["wavenumbers"], data["class_names"], len(data["class_names"]), spectra.shape[1]


def train(cfg: dict) -> None:
    device = get_device()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    loader, wavenumbers, class_names, n_classes, spectrum_len = load_data(cfg["data_path"], cfg["batch_size"])
    cfg["n_classes"] = n_classes
    cfg["spectrum_len"] = spectrum_len
    generator = Generator(cfg["latent_dim"], n_classes, spectrum_len, cfg["label_emb_dim"]).to(device)
    discriminator = Discriminator(n_classes, spectrum_len, cfg["label_emb_dim"]).to(device)
    ce_loss = nn.CrossEntropyLoss()
    opt_g = optim.Adam(generator.parameters(), lr=cfg["lr_g"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_d = optim.Adam(discriminator.parameters(), lr=cfg["lr_d"], betas=(cfg["beta1"], cfg["beta2"]))
    sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=cfg["epochs"])
    sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=cfg["epochs"])
    history = {"d_loss": [], "g_loss": [], "gp": [], "w_dist": []}
    fixed_z = torch.randn(n_classes * 4, cfg["latent_dim"], device=device)
    fixed_labels = torch.arange(n_classes, device=device).repeat_interleave(4)

    for epoch in range(1, cfg["epochs"] + 1):
        d_losses, g_losses, gp_vals, w_dists = [], [], [], []
        for i, (real_spectra, real_labels) in enumerate(loader):
            batch_size = real_spectra.size(0)
            real_spectra = real_spectra.to(device)
            real_labels = real_labels.to(device)
            z = torch.randn(batch_size, cfg["latent_dim"], device=device)
            fake_spectra = generator(z, real_labels).detach()
            d_real, logits_real = discriminator.forward_with_aux(real_spectra, real_labels)
            d_fake, logits_fake = discriminator.forward_with_aux(fake_spectra, real_labels)
            gp = gradient_penalty(discriminator, real_spectra, fake_spectra, real_labels, device)
            ac_loss_d = ce_loss(logits_real, real_labels) + ce_loss(logits_fake, real_labels)
            d_loss = d_fake.mean() - d_real.mean() + cfg["gp_lambda"] * gp + cfg["ac_lambda"] * ac_loss_d
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            d_losses.append(float(d_loss.item()))
            gp_vals.append(float(gp.item()))
            w_dists.append(float((d_real.mean() - d_fake.mean()).item()))

            if (i + 1) % cfg["n_critic"] == 0:
                z = torch.randn(batch_size, cfg["latent_dim"], device=device)
                fake_spectra = generator(z, real_labels)
                d_fake_g, logits_fake_g = discriminator.forward_with_aux(fake_spectra, real_labels)
                g_loss = -d_fake_g.mean() + cfg["ac_lambda"] * ce_loss(logits_fake_g, real_labels)
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                g_losses.append(float(g_loss.item()))

        history["d_loss"].append(float(np.mean(d_losses)))
        history["g_loss"].append(float(np.mean(g_losses) if g_losses else 0.0))
        history["gp"].append(float(np.mean(gp_vals)))
        history["w_dist"].append(float(np.mean(w_dists)))
        sched_g.step()
        sched_d.step()

        if epoch % cfg["plot_every"] == 0 or epoch == 1:
            with torch.no_grad():
                samples = generator(fixed_z, fixed_labels).cpu().numpy()
            plot_samples(samples, fixed_labels.cpu().numpy(), wavenumbers, class_names, f"Generated Spectra - Epoch {epoch}", output_dir / f"samples_epoch_{epoch:04d}.png")
        if epoch % cfg["save_every"] == 0:
            torch.save({"epoch": epoch, "G_state": generator.state_dict(), "D_state": discriminator.state_dict(), "opt_G_state": opt_g.state_dict(), "opt_D_state": opt_d.state_dict(), "cfg": cfg}, output_dir / f"checkpoint_epoch_{epoch:04d}.pt")

    torch.save({"epoch": cfg["epochs"], "G_state": generator.state_dict(), "D_state": discriminator.state_dict(), "cfg": cfg}, output_dir / "final_model.pt")
    plot_gan_losses(history, output_dir / "loss_curves.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the cWGAN-GP spectral generator.")
    for key, value in DEFAULTS.items():
        arg = f"--{key}"
        if isinstance(value, bool):
            parser.add_argument(arg, action="store_true", default=value)
        elif value is None:
            parser.add_argument(arg, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train({**DEFAULTS, **vars(args)})
