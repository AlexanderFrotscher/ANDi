# Normative-Diffusion

This repository contains multiple programs for detecting anomalies in brain MRI.  The newly implemented algorithm is based on Denoising Diffusion Probabilistic Models. The algorithm calculates differences between ground truth transition means and the predicted mean from the normative model. These differences correspond to the anomalies. Already published methods that include a DAE and theresholding approach can be found in the baseline folder.

## Usage

- Create an evironment with the .yml file
- Train with diffusion_cfg.py
- Evaluate with eval_3D.py

All "command line arguments / flags" are hardcoded in the code and can be modified there. The script diffusion_cfg.py trains the DDPM and contains the setup for the diffusion process as well as the new algorithms. The eval_3D script evaluates the method on volumes. All dataloader and preprocessing is found in utils.py and modules.py contains the U-Net architecture.

## Appendix

This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
