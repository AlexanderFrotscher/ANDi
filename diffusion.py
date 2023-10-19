__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

from utils import *

class Diffusion:
    def __init__(self, noise_steps=1000, img_size=128, device="cuda"):
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

    def noise_images(self, x, t, coarse=False, pyramid=False):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        if coarse == True:
            noise = coarse_noise(x.shape[0], x.shape[1], x.device)
        elif pyramid == True:
            noise = pyramid_noise_like(x.shape[0], x.shape[1], x.device)
        else:
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

    def ddpm_mean_t(self, x, t, predicted_noise=None, x_0=None):
        if predicted_noise == None and x_0 == None:
            print("Either noise or x_0 have to be given to calculate x_t-1.")
            exit(1)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
        if x_0 == None:
            pred_x0 = (
                1 / torch.sqrt(alpha_hat) * x
                - torch.sqrt((1 - alpha_hat) / (alpha_hat)) * predicted_noise
            )
            pred_x0 = pred_x0.clamp(-1, 1)
            # pred_x0 = clamp_to_spatial_quantile(pred_x0,0.99)
            x_0 = pred_x0
        w0 = torch.sqrt(alpha_hat_minus_one) * beta / (1 - alpha_hat)
        wt = torch.sqrt(alpha) * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
        return w0 * x_0 + wt * x

    def sample(
        self, model, n, labels, channels, cfg_scale=3, coarse=False, pyramid=False
    ):  # cfg scale determines the influence of the conditional model
        model.eval()
        with torch.no_grad():
            if coarse == True:
                x = coarse_noise(n, channels, self.device)
            elif pyramid == True:
                x = pyramid_noise_like(n, channels, self.device)
            else:
                x = torch.randn((n, channels, self.img_size, self.img_size)).to(
                    self.device
                )
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    if coarse == True:
                        noise = coarse_noise(n, channels, self.device)
                    elif pyramid == True:
                        noise = pyramid_noise_like(n, channels, self.device)
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(beta) * noise
                # x = self.ddpm_mu_t_2(x, predicted_noise, t) + torch.sqrt(beta) * noise
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
            t = (torch.ones(num_images) * timestemp - 1).long().to(self.device)
            x_t, noise = self.noise_images(images, t)
            xts[:, timestemp - 1] = x_t
            for i in tqdm(reversed(range(1, timestemp - 1)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                mean = self.ddpm_mean_t(xts[:, i + 1], t, x_0=images)
                xts[:, i] = mean
            xts[:, 0] = images

            for i in tqdm(reversed(range(1, timestemp)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t = xts[:, i]
                x_tm1 = xts[:, i - 1]
                predicted_noise = model(x_t, t)
                mu_t = self.ddpm_mu_t(x_t, predicted_noise, t)
                z_t = x_tm1 - mu_t
                zs[:, i - 1] = z_t
        return zs

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

            for i in tqdm(reversed(range(1, timestemp)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t = xts[:, i]
                x_tm1 = xts[:, i - 1]
                predicted_noise = model(x_t, t)
                mu_t = self.ddpm_mu_t(x_t, predicted_noise, t)
                z_t = x_tm1 - mu_t
                zs[:, i - 1] = z_t
        return zs

    def dpm_differences(self, model, images, start=100, stop=None, pyramid=False):
        if stop == None:
            stop = self.noise_steps
        if start == 0:
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            # First, sample from the forward process
            xts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            zs = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(images, t, pyramid=pyramid)
                xts[:, i - start] = x_t

            # calculate the differences
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t = xts[:, i - start]
                predicted_noise = model(x_t, t)
                mu_t = self.ddpm_mu_t(x_t, predicted_noise, t)
                mean = self.ddpm_mean_t(x_t, t, x_0=images)
                # what was supposed to be predicted and what is predicted
                z_t = (mean - mu_t) ** 2
                zs[:, i - start] = z_t
        return zs

    def skip_differences(
        self, model, images, start=100, stop=None, skip=25, pyramid=False
    ):
        if stop is None:
            stop = self.noise_steps
        if start == 0:
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            # First, sample from the forward process
            xts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            zs = torch.zeros(
                (
                    num_images,
                    int(((stop - start) / skip)),
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(images, t, pyramid=pyramid)
                xts[:, i - start] = x_t

            correct_chain = xts[:, -1]
            predicted_chain = xts[:, -1]

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                predicted_noise = model(predicted_chain, t)
                predicted_chain = self.ddpm_mu_t(predicted_chain, predicted_noise, t)
                correct_chain = self.ddpm_mean_t(correct_chain, t, x_0=images)
                if i % skip == 0 or i == 1:
                    z_t = (correct_chain - predicted_chain) ** 2
                    zs[:, int((i - start) / skip)] = z_t
                    predicted_chain = xts[:, i - start]
                    correct_chain = xts[:, i - start]
        return zs

    def differences_noise(self, model, images, start=100, stop=None, pyramid=False):
        if stop is None:
            stop = self.noise_steps
        if start == 0:
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            # First, sample from the forward process
            xts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            noises = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            zs = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(images, t, pyramid=pyramid)
                xts[:, i - start] = x_t
                noises[:, i - start] = noise

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t = xts[:, i - start]
                true_noise = noises[:, i - start]
                predicted_noise = model(x_t, t)
                z_t = (predicted_noise - true_noise) ** 2
                zs[:, i - start] = z_t
        return zs

    def multiple_latents(self, model, images, start=100, stop=None, num_latents = 3, pyramid=False):
        if stop == None:
            stop = self.noise_steps
        if start == 0:
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            # First, sample from the forward process
            xts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    num_latents,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            zs = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            #sigma_t = torch.zeros(
            #    (
            #        num_images,
            #        stop - start,
            #        images.shape[1],
            #        images.shape[2],
            #        images.shape[3],
            #    )
            #).to(self.device)
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                for j in range(num_latents):
                    x_t, noise = self.noise_images(images, t, pyramid=pyramid)
                    xts[:, i - start, j] = x_t

            # calculate the differences
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                z_t_list = []
                for j in range(num_latents):
                    x_t = xts[:, i - start, j]
                    predicted_noise = model(x_t, t)
                    mu_t = self.ddpm_mu_t(x_t, predicted_noise, t)
                    mean = self.ddpm_mean_t(x_t, t, x_0=images)
                    # what was supposed to be predicted and what is predicted
                    z_t = (mean - mu_t) ** 2
                    z_t_list.append(z_t)
                zt = torch.cat(z_t_list,dim=0)
                zs[:, i - start] = torch.mean(zt,dim=0,keepdim=True)
                #sigma_t[:, i - start] = torch.sqrt(torch.var(zt,dim=0,keepdim=True))
        return zs

    # initial idea of using z_t -> does not work well
    def guide_restoration(self, model, xts, zs, cfg_scale=1.5, noise_scale=0.5):
        model.eval()
        num_steps = xts.shape[1]
        num_images = zs.shape[0]
        with torch.no_grad():
            x = xts[:, -1]
            for i in tqdm(reversed(range(1, num_steps)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                predicted_noise = model(x, t)
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
                    noise_scale * noise + (1 - noise_scale) * zs[:, i - 1]
                )
        model.train()
        return x

    def ano_ddpm(self, model, images, num_steps):
        model.eval()
        num_images = images.shape[0]
        with torch.no_grad():
            t = (torch.ones(num_images) * num_steps).long().to(self.device)
            x, noise = self.noise_images(images, t)
            for i in tqdm(reversed(range(1, num_steps)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                predicted_noise = model(x, t)
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(beta) * noise
        model.train()
        return x
