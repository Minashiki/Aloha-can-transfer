"""Diffusion policy (image behavior cloning).

The Gaussian diffusion loss, conditional UNet, DDIM sampler (eta = 0), ViT
encoder, and random-shift augmentation are adapted from DPPO. See
``THIRD_PARTY_LICENSES``. Call ``DiffusionPolicy.sample(obs, noise)`` to map
an initial noise tensor to an action chunk.
"""

from aloha_pp.policy.policy import DiffusionPolicy

__all__ = ["DiffusionPolicy"]
