import math
import numpy as np
import torch

from functools import partial

def create_model(cfg, device, aabb=None, near_far=None):
    # initialization
    model = None; reso_cur = None

    if "Hex" in cfg.model.model_name:
        from .hexplane.src.HexPlane import HexPlane
        from .hexplane.src.HexPlane_Slim import HexPlane_Slim
        from .hexplane.render.util.util import N_to_reso
        
        reso_cur = N_to_reso(cfg.model.N_voxel_init, aabb, cfg.model.nonsquare_voxel)

        if cfg.systems.ckpt is not None:
            model = torch.load(cfg.systems.ckpt, map_location=device)
        else:
            # There are two types of upsampling: aligned and unaligned.
            # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
            # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
            if cfg.model.upsampling_type == "aligned":
                reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]
            model = eval(cfg.model.model_name)(aabb, 
                                            reso_cur, 
                                            device, 
                                            cfg.model.time_grid_init, 
                                            near_far, 
                                            **cfg.model)
    
    elif "SynergyNeRF" in cfg.model.model_name:
        if not len(cfg.model.model_name.split("_")) > 1:
            from .SynergyNeRF.src.SynergyNeRF import SynergyNeRF
            from .SynergyNeRF.render.util.util import N_to_reso
            reso_cur = N_to_reso(cfg.model.N_voxel_init, aabb, cfg.model.nonsquare_voxel)
            if cfg.systems.ckpt is not None:
                model = torch.load(cfg.systems.ckpt, map_location=device)
            else:
                # There are two types of upsampling: aligned and unaligned.
                # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
                # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
                if cfg.model.upsampling_type == "aligned":
                    reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]
                model = eval(cfg.model.model_name)(aabb, 
                                                reso_cur, 
                                                device, 
                                                cfg.model.time_grid_init, 
                                                near_far, 
                                                **cfg.model)
        elif "disentangled" in cfg.model.model_name:
            from .SynergyNeRF_disentangled.src.SynergyNeRF import SynergyNeRF as SynergyNeRF_disentangled
            from .SynergyNeRF_disentangled.render.util.util import N_to_reso
            reso_cur = N_to_reso(cfg.model.N_voxel_init, aabb, cfg.model.nonsquare_voxel)
            if cfg.systems.ckpt is not None:
                model = torch.load(cfg.systems.ckpt, map_location=device)
            else:
                # There are two types of upsampling: aligned and unaligned.
                # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
                # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
                if cfg.model.upsampling_type == "aligned":
                    reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]
                model = eval(cfg.model.model_name)(aabb, 
                                                reso_cur, 
                                                device, 
                                                cfg.model.time_grid_init, 
                                                near_far, 
                                                **cfg.model)
        else:
            NotImplementedError("[create_model] Invalid model name")
    else:
        NotImplementedError("[create_model] Invalid model name")
            

    return {"model" : model, 
            "reso_cur" : reso_cur}
            

def create_trainer_and_render(cfg, model, train_dataset, val_dataset, tb_writer, logdir, device, **kwargs):
    render_disentangled = None
    
    if "Hex" in cfg.model.model_name:
        from .hexplane.render.trainer import Trainer
        from .hexplane.render.render import render, render_trajectory, vis_planes
        reso_cur = kwargs.get("reso_cur", None); assert reso_cur is not None, "Invalid current resolution"
        trainer = Trainer(model, cfg, reso_cur, train_dataset, val_dataset, tb_writer, logdir, device)

    elif "SynergyNeRF" in cfg.model.model_name:
        if not len(cfg.model.model_name.split("_")) > 1:
            from .SynergyNeRF.render.trainer import Trainer
            from .SynergyNeRF.render.render import render, render_trajectory, vis_planes
            reso_cur = kwargs.get("reso_cur", None); assert reso_cur is not None, "Invalid current resolution"
            trainer = Trainer(model, cfg, reso_cur, train_dataset, val_dataset, tb_writer, logdir, device)
        elif "disentangled" in cfg.model.model_name:
            from .SynergyNeRF_disentangled.render.trainer import Trainer
            from .SynergyNeRF_disentangled.render.render import render, render_trajectory, vis_planes
            reso_cur = kwargs.get("reso_cur", None); assert reso_cur is not None, "Invalid current resolution"
            trainer = Trainer(model, cfg, reso_cur, train_dataset, val_dataset, tb_writer, logdir, device)
        else:
            NotImplementedError("[create_trainer] Invalid model name")
    else:
        NotImplementedError("[create_trainer] Invalid model name")

    return {"trainer" : trainer, 
            "render_fn" : render, 
            "render_trajectory_fn" : render_trajectory,
            "vis_planes_fn" : vis_planes,
            "render_disentangled_fn" : render_disentangled if render_disentangled is not None else None
            }