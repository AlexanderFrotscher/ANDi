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

from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim

import wandb
from diffusion import *
from modules import *
from utils import *

logging.basicConfig(format="%(message)s", level=logging.INFO)
os.environ["WANDB__SERVICE_WAIT"] = "300"

def train(args):
    make_dicts(args.run_name)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    dataloader = Brats21(args, preload=True)
    steps_per_epoch = int(np.ceil(len(dataloader.dataset) / args.batch_size))
    number_of_steps = steps_per_epoch * args.epochs
    model = UNet(
        c_in=args.channels,
        c_out=args.channels,
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

    # model = torch.compile(model)
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
            images = (images * 2) - 1  # normalization
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t, simplex=True, pyramid=False)
            # if np.random.random() < 0.1:
            #    labels = None
            # predicted_noise = model(x_t, t, labels)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            ema.step_ema(ema_model, model)
            wandb.log({"MSE": loss.item()})

        if epoch > 120 and epoch % 8 == 0 and accelerator.is_main_process:
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
                ema_model,
                n=n,
                labels=labels,
                channels=args.channels,
                cfg_scale=0,
                pyramid=False,
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
    args.run_name = "Brats128_simplex"
    args.epochs = 233
    args.batch_size = 128
    args.image_size = 128
    args.channels = 4
    args.dataset_path = (
        "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    )
    args.start_lr = 2e-5
    args.target_lr = 7e-5
    args.path_to_csv = "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data/splits/scans_train.csv"
    args.train_continue = False
    #args.current_model = "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/Brats128_pyramid/320_ckpt.pt"
    #args.current_ema = "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/Brats128_pyramid/320_ema_ckpt.pt"
    #args.current_opt = "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/Brats128_pyramid/320_optim.pt"
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )

    wandb.init(entity="team-frotscher", project=args.run_name, config=args)

    train(args)


if __name__ == "__main__":
    main()
