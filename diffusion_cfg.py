__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import argparse
import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch import optim
from tqdm import tqdm

import wandb
from modules import *
from utils import *


logging.basicConfig(format="%(message)s", level=logging.INFO)


class Diffusion:
    def __init__(self, noise_steps=1000, img_size=32, device="cuda"):
        self.noise_steps = noise_steps

        self.beta = self.linear_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def linear_noise_schedule(self):
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def cosine_beta_schedule(self, s=0.008):
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(
        self, model, n, labels, cfg_scale=3
    ):  # cfg scale determines the influence of the conditional model
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
        model.train()
        x = (
            x.clamp(-1, 1) + 1
        ) / 2  # other code uses permute, why, is clamping the sampled image ok?
        x = (x * 255).type(torch.uint8)
        return x

    def ddim_sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(
                reversed(np.linspace(1, 1000 - 1, 250).astype(int)), position=0
            ):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                # the notation in the ddim paper is different -> alpha_hat is just alpha there
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                x = torch.sqrt(alpha_hat_minus_one) * (
                    (x - torch.sqrt(1 - alpha_hat_minus_one) * predicted_noise)
                    / torch.sqrt(alpha_hat)
                ) + (torch.sqrt(1 - alpha_hat_minus_one) * predicted_noise)
        model.train()
        x = (
            x.clamp(-1, 1) + 1
        ) / 2  # other code uses permute, why, is clamping the sampled image ok?
        x = (x * 255).type(torch.uint8)
        return x

    def create_latent(self, x, model):
        logging.info(f"Creating {x.shape[0]} new latents....")
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(1, self.noise_steps), position=0):
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, None)
                alpha_hat_plus_one = self.alpha_hat[t + 1][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                x = torch.sqrt(alpha_hat_plus_one) * (
                    (x - torch.sqrt(1 - alpha_hat) * predicted_noise)
                    / torch.sqrt(alpha_hat)
                ) + (torch.sqrt(alpha_hat_plus_one) * predicted_noise)
        return x

    def reconstruction(self, model, latents, labels, cfg_scale=0):
        logging.info(f"Sampling {latents.shape[0]} new images....")
        model.eval()
        with torch.no_grad():
            for i in tqdm(
                reversed(np.linspace(1, 1000 - 1, 250).astype(int)), position=0
            ):
                t = (torch.ones(latents.shape[0]) * i).long().to(self.device)
                predicted_noise = model(latents, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                x = torch.sqrt(alpha_hat_minus_one) * (
                    (x - torch.sqrt(1 - alpha_hat_minus_one) * predicted_noise)
                    / torch.sqrt(alpha_hat)
                ) + (torch.sqrt(1 - alpha_hat_minus_one) * predicted_noise)
        model.train()
        x = (
            x.clamp(-1, 1) + 1
        ) / 2  # other code uses permute, why, is clamping the sampled image ok?
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    make_dicts(args.run_name)
    accelerator = Accelerator(find_unused_parameters=True)
    device = accelerator.device
    dataloader = cifar_10(args)
    model = UNet_conditional(num_classes=args.num_classes, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    wandb.watch(model, log="all")
    model, ema_model, optimizer, dataloader = accelerator.prepare(
        model, ema_model, optimizer, dataloader
    )

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(
                device
            )  # every picture gets one timestep in one epoch
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            ema.step_ema(ema_model, model)

            wandb.log({"MSE": loss.item()})

        if epoch % 10 == 0:
            labels = torch.arange(args.num_classes).long().to(device)
            torch.save(
                model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt")
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join("models", args.run_name, f"optim.pt"),
            )
            ema_sampled_images = diffusion.sample(
                ema_model, n=len(labels), labels=labels
            )
            save_images(
                ema_sampled_images,
                os.path.join("results", args.run_name, f"{epoch}_ema.jpg"),
            )
            torch.save(
                ema_model.state_dict(),
                os.path.join("models", args.run_name, f"ema_ckpt.pt"),
            )
            torch.save(
                ema_model.state_dict(), os.path.join(wandb.run.dir, f"ema_ckpt.pt")
            )
            wandb.save(os.path.join(wandb.run.dir, "ema_ckpt.pt"))

            ddim_ema = diffusion.ddim_sample(ema_model, n=len(labels), labels=labels)
            example_images = wandb.Image(upload_images(ema_sampled_images))
            ddim_images = wandb.Image(upload_images(ddim_ema))
            wandb.log({"EMA-DDPM": example_images, "EMA-DDIM": ddim_images})


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "diffusion_test"
    args.epochs = 101
    args.batch_size = 128
    args.image_size = 32
    args.num_classes = 10
    args.dataset_path = ""
    args.lr = 3e-4
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )

    wandb.init(entity="team-frotscher", project=args.run_name, config=args)

    train(args)


if __name__ == "__main__":
    main()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
