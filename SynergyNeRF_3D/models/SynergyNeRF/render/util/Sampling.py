import numpy as np
import torch


def GM_function(x, gamma):
    return x**2 / (x**2 + gamma**2)


def GM_Resi(x, y, gamma):
    return 1 / 3 * torch.norm(GM_function(x - y, gamma), p=1, dim=-1)


def cal_n_samples(reso, step_ratio=0.5, max_value=1e+6, min_value=128):
    n_samples = int(np.linalg.norm(reso) / step_ratio)

    # check threshold
    n_samples = min(max_value, n_samples)
    n_samples = max(min_value, n_samples)
    
    return n_samples

