__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import logging

import yaml
from accelerate import Accelerator
from dae_unet import *
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from utils import *

logging.basicConfig(format="%(message)s", level=logging.INFO)


def loss_f(y, predictions, mask):
    return (torch.pow(predictions - y, 2) * mask.float()).mean()


def train(conf):
    make_dicts(conf["run_name"])
    accelerator = Accelerator()
    device = accelerator.device
    dataloader = Brats21(conf)
    model = UNet(
        conf["channels"],
        conf["channels"],
        conf["depth"],
        conf["wf"],
        padding=conf["padding"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf["lr"],
        amsgrad=conf["amsgrad"],
        weight_decay=conf["weight_decay"],
    )

    if conf["train_continue"] == True:
        ckpt = torch.load(conf["model"])
        model.load_state_dict(ckpt)
        ckpt = torch.load(conf["opt"])
        optimizer.load_state_dict(ckpt)

    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=conf["T_max"])
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dataloader
    )

    for epoch in range(conf["epochs"]):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader) if accelerator.is_main_process else dataloader
        for i, (images) in enumerate(pbar):
            mask = images.sum(dim=1, keepdim=True) > 0.01
            my_noise = coarse_noise(
                images.shape[0],
                images.shape[1],
                images.device,
                noise_size=conf["noise_size"],
                noise_std=conf["noise_std"],
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
        if epoch % conf["save_ckpt"] == 0 and accelerator.is_main_process:
            my_model = accelerator.unwrap_model(model)
            accelerator.save(
                my_model.state_dict(),
                os.path.join("models", conf["run_name"], f"{epoch}_ckpt.pt"),
            )
            accelerator.save(
                optimizer.state_dict(),
                os.path.join("models", conf["run_name"], f"{epoch}_optim.pt"),
            )


def main():
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )
    with open("../conf/train_dae.yml", "r") as file_object:
        conf = yaml.load(file_object, Loader=yaml.SafeLoader)

        wandb.init(entity="team-frotscher", project=conf["run_name"], config=conf)
        train(conf)


if __name__ == "__main__":
    main()
