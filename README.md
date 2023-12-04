# ANDi

This repository contains the code for the "Unsupervised Anomaly Detection using Aggregated Normative Diffusion" publication. Aggregated Normative Diffusion (ANDi) is made for detecting anomalies in brain MRI and is based on Denoising Diffusion Probabilistic Models. The algorithm calculates differences between ground truth transition means and the predicted mean from the normative model. These differences correspond to the anomalies. Already published methods that include a DAE and theresholding approach can be found in the baseline folder.


## Appendix

This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
