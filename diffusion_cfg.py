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
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim
from tqdm import tqdm

import wandb
from modules import *
from utils import *

logging.basicConfig(format="%(message)s", level=logging.INFO)


class Diffusion:
    def __init__(self, noise_steps=1000, img_size=64, device="cuda"):
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
        x = torch.linspace(1, self.noise_steps)
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

    def ddpm_mu_t(self, x, predicted_noise, t):
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        return (
            1
            / torch.sqrt(alpha)
            * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
        )

    def sample(
        self, model, n, labels, channels, cfg_scale=3
    ):  # cfg scale determines the influence of the conditional model
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def ddim_sample(
        self, model, n, labels, channels, cfg_scale=3
    ):  # not working at all
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(
                reversed(
                    np.linspace(1, self.noise_steps, 250, endpoint=False).astype(int)
                ),
                position=0,
            ):
                t = (torch.ones(n) * i).long().to(self.device)
                t_m1 = (torch.ones(n) * (i - 1)).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                # the notation in the ddim paper is different -> alpha_hat is just alpha there
                alpha_hat_minus_one = self.alpha_hat[t_m1][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                x = torch.sqrt(alpha_hat_minus_one) * (
                    (x - torch.sqrt(1 - alpha_hat_minus_one) * predicted_noise)
                    / torch.sqrt(alpha_hat)
                ) + (torch.sqrt(1 - alpha_hat_minus_one) * predicted_noise)
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def dpm_encoder(self, model, images, timestemp=None):
        if timestemp is None:
            timestemp = self.noise_steps
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            # First, sample from the forward process by drawing only once
            xts = torch.zeros(
                (
                    num_images,
                    timestemp,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            zs = torch.zeros(
                (
                    num_images,
                    timestemp - 1,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            t = (torch.ones(num_images) * timestemp-1).long().to(self.device)
            x_t, noise = self.noise_images(images, t)
            xts[:, timestemp - 1] = x_t
            for i in tqdm(reversed(range(1, timestemp-1)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                t_m1 = (torch.ones(num_images) * (i - 1)).long().to(self.device)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t_m1][:, None, None, None]
                w0 = torch.sqrt(alpha_hat_minus_one) * beta / (1 - alpha_hat)
                wt = torch.sqrt(alpha) * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                mean = w0 * images + wt * xts[:, i]
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                xts[:, i - 1] = mean + var.sqrt() * torch.rand_like(images)
            xts[:, 0] = images
            # generate the latents
            for i in tqdm(reversed(range(1, timestemp)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t = xts[:, i]
                x_tm1 = xts[:, i - 1]
                predicted_noise = model(x_t, t, None)
                mu_t = self.ddpm_mu_t(x_t, predicted_noise, t)
                beta = self.beta[t][:, None, None, None]
                z_t = (x_tm1 - mu_t) / torch.sqrt(beta)
                zs[:, i - 1] = z_t
        return xts, zs

    def dpm_inversion(self, model, images, timestemp=None):
        if timestemp is None:
            timestemp = self.noise_steps
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            # First, sample from the forward process
            xts = torch.zeros(
                (
                    num_images,
                    timestemp,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            zs = torch.zeros(
                (
                    num_images,
                    timestemp - 1,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            for i in tqdm(reversed(range(1, timestemp)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(images, t)
                xts[:, i] = x_t
            xts[:, 0] = images

            # generate the latents
            for i in tqdm(reversed(range(1, timestemp)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t = xts[:, i]
                x_tm1 = xts[:, i - 1]
                predicted_noise = model(x_t, t, None)
                mu_t = self.ddpm_mu_t(x_t, predicted_noise, t)
                beta = self.beta[t][:, None, None, None]
                z_t = (x_tm1 - mu_t) / torch.sqrt(beta)
                zs[:, i - 1] = z_t
                if i > 1:
                    # avoid accumulation error
                    x_tm1 = mu_t + (torch.sqrt(beta) * z_t)
                    xts[:, i - 1] = x_tm1
        return xts, zs

    def guide_restoration(self, model, xts, zs, cfg_scale=1.5, noise_scale=0.5):
        logging.info(f"Starting healing process....")
        model.eval()
        num_steps = xts.shape[1]
        num_images = zs.shape[0]
        with torch.no_grad():
            x = xts[:, -1]
            for i in tqdm(reversed(range(1, num_steps)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                predicted_noise = model(x, t, None)
                if cfg_scale > 0:
                    sample_noise = model(xts[:, i], t, None)
                    predicted_noise = torch.lerp(
                        sample_noise, predicted_noise, cfg_scale
                    )
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(beta) * (
                    noise_scale * noise + (1 - noise_scale) * zs[:, i-1]
                )
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x

    def create_mask(self, zs, timestep=None, threshold=2.58):  # bonferoni 4.374
        if timestep is None:
            timestep = self.noise_steps
        zs_mean = torch.mean(zs, dim=1)
        zs_mean = (torch.abs(zs_mean)) * np.sqrt(timestep)
        zs_mean = torch.mean(zs_mean, dim=1)
        zs_mean[zs_mean < threshold] = 0
        zs_mean[zs_mean != 0] = 1
        return zs_mean.type(torch.uint8)


def create_difference(images, predictions, threshold=0.2):
    masks = torch.abs(images - predictions)
    masks = torch.mean(masks, dim=1)
    masks[masks < threshold] = 0
    masks[masks != 0] = 1
    return masks.type(torch.uint8)


def train(args):
    make_dicts(args.run_name)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    dataloader = Brats20(args, preload=True)
    steps_per_epoch = int(np.ceil(len(dataloader.dataset) / args.batch_size))
    number_of_steps = steps_per_epoch * args.epochs
    model = UNet_conditional(
        c_in=args.channels,
        c_out=args.channels,
        num_classes=args.num_classes,
        device=device,
        img_size=args.image_size,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=1
    )  # scheduler multiplies base lr of optimizer -> lr = 1

    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    # wandb.watch(model, log="all")

    if args.train_continue == True:
        args.target_lr = args.start_lr
        ckpt = torch.load(args.current_model)
        model.load_state_dict(ckpt)
        ckpt = torch.load(args.current_opt)
        optimizer.load_state_dict(ckpt)
        ckpt = torch.load(args.current_ema)
        ema_model.load_state_dict(ckpt)
        ema_model.eval().requires_grad_(False)

    scheduler = LRWarmupCosineDecay(
        optimizer,
        int(0.05 * number_of_steps),
        number_of_steps,
        args.start_lr,
        args.target_lr,
    )
    model, ema_model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, ema_model, optimizer, scheduler, dataloader
    )

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images) in enumerate(pbar):
            # images = images.to(device)
            # labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(
                device
            )  # every picture gets one timestep in one epoch
            x_t, noise = diffusion.noise_images(images, t)
            # if np.random.random() < 0.1:
            #    labels = None
            # predicted_noise = model(x_t, t, labels)
            predicted_noise = model(x_t, t, None)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            ema.step_ema(ema_model, model)
            wandb.log({"MSE": loss.item()})

        if epoch % 8 == 0 and accelerator.is_main_process:
            my_model = accelerator.unwrap_model(model)
            my_ema_model = accelerator.unwrap_model(ema_model)
            # labels = torch.arange(args.num_classes).long().to(device)
            labels = None
            n = 5
            accelerator.save(
                my_model.state_dict(),
                os.path.join("models", args.run_name, f"{epoch}_ckpt.pt"),
            )
            accelerator.save(
                optimizer.state_dict(),
                os.path.join("models", args.run_name, f"{epoch}_optim.pt"),
            )
            ema_sampled_images = diffusion.sample(
                ema_model, n=n, labels=labels, channels=args.channels, cfg_scale=0
            )
            save_images(
                ema_sampled_images,
                os.path.join("results", args.run_name, f"{epoch}_ema.jpg"),
                mode="L",
            )
            accelerator.save(
                my_ema_model.state_dict(),
                os.path.join("models", args.run_name, f"{epoch}_ema_ckpt.pt"),
            )
            example_images = wandb.Image(upload_images(ema_sampled_images, mode="L"))
            wandb.log({"EMA-DDPM": example_images})


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "BraTS21_5"
    args.epochs = 161
    args.batch_size = 20
    args.image_size = 64
    args.channels = 4
    args.num_classes = None  # 116
    args.dataset_path = (
        "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data"
    )
    # args.dataset_path = './data/BraTS20'
    args.start_lr = 2e-5
    args.target_lr = 1e-4
    args.path_to_csv = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/scans_train.csv"
    # args.path_to_csv = './data/survival_info_02.csv'
    args.train_continue = False
    args.current_model = (
        "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/BraTS21_3/64_ckpt.pt"
    )
    args.current_ema = "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/BraTS21_3/64_ema_ckpt.pt"
    args.current_opt = "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/BraTS21_3/64_optim.pt"
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )

    wandb.init(entity="team-frotscher", project=args.run_name, config=args)

    train(args)


if __name__ == "__main__":
    main()
