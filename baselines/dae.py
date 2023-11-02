__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse
import logging
import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from accelerate import Accelerator
from dae_unet import *
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from utils import *

logging.basicConfig(format="%(message)s", level=logging.INFO)


def loss_f(y, predictions, mask):
    return (torch.pow(predictions - y, 2) * mask.float()).mean()


def train(args):
    torch.manual_seed(0)
    make_dicts(args.run_name)
    accelerator = Accelerator()
    device = accelerator.device
    dataloader = Brats21(args, preload=True)
    model = UNet(args.channels, args.channels, depth=4, wf=6, padding=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.00001
    )

    if args.train_continue == True:
        ckpt = torch.load(args.current_model)
        model.load_state_dict(ckpt)
        ckpt = torch.load(args.current_opt)
        optimizer.load_state_dict(ckpt)

    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=100)
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dataloader
    )

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images) in enumerate(pbar):
            mask = images.sum(dim=1, keepdim=True) > 0.01
            my_noise = coarse_noise(
                images.shape[0],
                images.shape[1],
                images.device,
                image_size=images.shape[2],
            )
            my_noise *= mask
            noise_images = images.clone() + my_noise
            predicted_image = model(noise_images)
            loss = loss_f(images, predicted_image, mask)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            wandb.log({"loss": loss.item()})

        lr_scheduler.step()
        if epoch % 2 == 0 and accelerator.is_main_process:
            my_model = accelerator.unwrap_model(model)
            accelerator.save(
                my_model.state_dict(),
                os.path.join("models", args.run_name, f"{epoch}_ckpt.pt"),
            )
            accelerator.save(
                optimizer.state_dict(),
                os.path.join("models", args.run_name, f"{epoch}_optim.pt"),
            )


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DAE"
    args.epochs = 5
    args.batch_size = 16
    args.image_size = 128
    args.channels = 4
    args.dataset_path = (
        "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    )
    args.lr = 0.0001
    args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/scans_train.csv"
    args.train_continue = True
    args.current_model = (
        "/mnt/qb/work/baumgartner/bkc035/normative-diffusion/baselines/models/DAE/14_ckpt.pt"
    )
    args.current_opt = (
        "/mnt/qb/work/baumgartner/bkc035/normative-diffusion/baselines/models/DAE/14_optim.pt"
    )
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )
    wandb.init(entity="team-frotscher", project=args.run_name, config=args)

    train(args)


if __name__ == "__main__":
    main()
