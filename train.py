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

import hydra
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import DictConfig, OmegaConf
from torch import optim

import wandb
from diffusion import *
from modules import *
from utils import *

logging.basicConfig(format="%(message)s", level=logging.INFO)
os.environ["WANDB__SERVICE_WAIT"] = "300"


def train(conf):
    make_dicts(conf.run_name)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device

    if conf.simplex & conf.pyramid == True:  # both noise types result in Gaussian
        conf.simplex = False
        conf.pyramid = False


    dataloader = Brats21(
        conf, preload=True
    )  # This preloads all slices in the memory. Can only be used with enough memory
    steps_per_epoch = int(np.ceil(len(dataloader.dataset) / conf.batch_size))
    number_of_steps = steps_per_epoch * conf.epochs
    model = UNet(
        c_in=conf.channels,
        c_out=conf.channels,
        device=device,
        img_size=conf.size,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=1
    )  # scheduler multiplies base lr of optimizer -> lr = 1

    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=conf.size, beta_start=conf.beta_start, beta_end=conf.beta_end, device=device)
    ema = EMA(conf.ema_decay)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    # wandb.watch(model, log="all")

    if conf.train_continue == True:
        conf.target_lr = conf.start_lr
        ckpt = torch.load(conf.model)
        model.load_state_dict(ckpt)
        ckpt = torch.load(conf.opt)
        optimizer.load_state_dict(ckpt)
        ckpt = torch.load(conf.ema)
        ema_model.load_state_dict(ckpt)
        ema_model.eval().requires_grad_(False)

    # model = torch.compile(model)
    scheduler = LRWarmupCosineDecay(
        optimizer,
        int(conf.warmup_steps * number_of_steps),
        number_of_steps,
        conf.start_lr,
        conf.target_lr,
    )
    model, ema_model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, ema_model, optimizer, scheduler, dataloader
    )

    for epoch in range(conf.epoch):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader) if accelerator.is_main_process else dataloader
        for i, (images) in enumerate(pbar):
            images = (images * 2) - 1  # normalization
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t, simplex=False, pyramid=False)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            ema.step_ema(ema_model, model)
            if accelerator.is_main_process:
                wandb.log({"MSE": loss.item()})
                pbar.set_description(f"Loss: {loss.item():.4f}")

        if epoch > conf.start_ckpt and epoch % conf.save_ckpt == 0 and accelerator.is_main_process:
            my_model = accelerator.unwrap_model(model)
            my_ema_model = accelerator.unwrap_model(ema_model)
            accelerator.save(
                my_model.state_dict(),
                os.path.join("models", conf.run_name, f"{epoch}_ckpt.pt"),
            )
            accelerator.save(
                optimizer.state_dict(),
                os.path.join("models", conf.run_name, f"{epoch}_optim.pt"),
            )
            ema_sampled_images = diffusion.sample(
                ema_model, n=conf.num_images, channels=conf.channels, pyramid=False, simplex=False
            )
            save_images(
                ema_sampled_images,
                os.path.join("results", conf.run_name, f"{epoch}_ema.jpg"),
                mode="L",
            )
            accelerator.save(
                my_ema_model.state_dict(),
                os.path.join("models", conf.run_name, f"{epoch}_ema_ckpt.pt"),
            )
            example_images = wandb.Image(upload_images(ema_sampled_images, mode="L"))
            wandb.log({"EMA-DDPM": example_images})

@hydra.main(version_base="1.3", config_name='train')
def main(conf : DictConfig):
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )
    wandb.init(entity="team-frotscher", project=conf.run_name, config=conf)
    train(conf)

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.run_name = "Brats128_simplex"
    # args.epochs = 233
    # args.batch_size = 128
    # args.image_size = 128
    # args.channels = 4
    # args.dataset_path = (
    #    "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    # )
    # args.start_lr = 2e-5
    # args.target_lr = 7e-5
    # args.path_to_csv = "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data/splits/scans_train.csv"
    # args.train_continue = False
    # args.current_model = "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/Brats128_pyramid/320_ckpt.pt"
    # args.current_ema = "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/Brats128_pyramid/320_ema_ckpt.pt"
    # args.current_opt = "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/Brats128_pyramid/320_optim.pt"


if __name__ == "__main__":
    main()
