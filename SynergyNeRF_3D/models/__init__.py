import math
import numpy as np
import torch

from functools import partial
from typing import List, Optional, Tuple, Union

def create_model(cfg, device, aabb=None, near_far=None):
    # initialization
    model = None; reso_cur = None

    if "TensorVM" in cfg.model.model_name:
        from .TensoRF.src.tensoRF import TensorVM, TensorVMSplit, TensorCP
        from .TensoRF.render.util.util import N_to_reso
        
        reso_cur = N_to_reso(cfg.model.N_voxel_init, aabb, cfg.model.nonsquare_voxel)

        if cfg.systems.ckpt is not None:
            model = torch.load(cfg.systems.ckpt, map_location=device)
        else:
            model = eval(cfg.model.model_name)(aabb, 
                                               reso_cur, 
                                               near_far, 
                                               device, 
                                               **cfg.model)
    
    elif "SynergyNeRF" in cfg.model.model_name:
        if not len(cfg.model.model_name.split("_")) > 1:
            from .SynergyNeRF.src.SynergyNeRF import SynergyNeRF as SynergyNeRF
            from .SynergyNeRF.render.util.util import N_to_reso
            reso_cur = N_to_reso(cfg.model.N_voxel_init, aabb, cfg.model.nonsquare_voxel)
            if cfg.systems.ckpt is not None:
                model = torch.load(cfg.systems.ckpt, map_location=device)
                model.fp16 = cfg.model.fp16
            else:
                # There are two types of upsampling: aligned and unaligned.
                # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
                # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
                if cfg.model.upsampling_type == "aligned":
                    reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]
                model = eval(cfg.model.model_name)(aabb, 
                                                reso_cur, 
                                                near_far, 
                                                device, 
                                                **cfg.model)
        else:
            if "disentangled" in cfg.model.model_name:
                from .SynergyNeRF_disentangled.src.SynergyNeRF import SynergyNeRF as SynergyNeRF_disentangled
                from .SynergyNeRF_disentangled.render.util.util import N_to_reso
                
                reso_cur = N_to_reso(cfg.model.N_voxel_init, aabb, cfg.model.nonsquare_voxel)
                if cfg.systems.ckpt is not None:
                    model = torch.load(cfg.systems.ckpt, map_location=device)
                    
                    if cfg.model.upsampling_type == "aligned":
                        reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]
                    self_model = eval(cfg.model.model_name)(aabb, 
                                                    reso_cur, 
                                                    near_far, 
                                                    device, 
                                                    **cfg.model)
                    
                    # set attributes
                    self_model.set_attributes(model)
                    model = self_model
                    
                else:
                    # There are two types of upsampling: aligned and unaligned.
                    # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
                    # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
                    if cfg.model.upsampling_type == "aligned":
                        reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]
                    model = eval(cfg.model.model_name)(aabb, 
                                                    reso_cur, 
                                                    near_far, 
                                                    device, 
                                                    **cfg.model)
            else:
                NotImplementedError("[create_model] Invalid model name")    
    else:
        NotImplementedError("[create_model] Invalid model name")
            

    return {"model" : model, 
            "reso_cur" : reso_cur}
            

def create_trainer_and_render(cfg, model, datasets_dict, tb_writer, logdir, device, **kwargs):
    render_disentangled = None; render_mesh = None
    renderer = None
    
    if "TensorVM" in cfg.model.model_name:
        from .TensoRF.render.trainer import Trainer
        from .TensoRF.render.render import render, render_trajectory, vis_planes
        reso_cur = kwargs.get("reso_cur", None); assert reso_cur is not None, "Invalid current resolution"
        trainer = Trainer(model, cfg, reso_cur, datasets_dict["train_dataset"], datasets_dict["val_dataset"], tb_writer, logdir, device)

    elif "SynergyNeRF" in cfg.model.model_name:
        if not len(cfg.model.model_name.split("_")) > 1:
            from .SynergyNeRF.render.trainer import Trainer
            from .SynergyNeRF.render.render import render, render_trajectory, vis_planes
            reso_cur = kwargs.get("reso_cur", None); assert reso_cur is not None, "Invalid current resolution"
            trainer = Trainer(model, cfg, reso_cur, datasets_dict["train_dataset"], datasets_dict["val_dataset"], tb_writer, logdir, device)

        else:
            if "disentangled" in cfg.model.model_name:
                from .SynergyNeRF_disentangled.render.trainer import Trainer
                from .SynergyNeRF_disentangled.render.render import render, render_trajectory, render_disentangled, vis_planes
                reso_cur = kwargs.get("reso_cur", None); assert reso_cur is not None, "Invalid current resolution"
                trainer = Trainer(model, cfg, reso_cur, datasets_dict["train_dataset"], datasets_dict["val_dataset"], tb_writer, logdir, device)
            else: NotImplementedError("[create_trainer] Invalid model name")
    
    else:
        NotImplementedError("[create_trainer] Invalid model name")

    return {"trainer" : trainer, 
            "render_fn" : render, 
            "render_trajectory_fn" : render_trajectory, 
            "vis_planes_fn" : vis_planes if vis_planes is not None else None,
            "render_disentangled_fn" : render_disentangled if render_disentangled is not None else None,
            "render_mesh_fn" : render_mesh if render_mesh is not None else None,
            "renderer" : renderer if renderer is not None else None
            }