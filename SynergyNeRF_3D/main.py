import numpy as np
import torch

from models import create_model, create_trainer_and_render
from utils.setup import *


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    
    # Config / Seed
    cfg = set_config(); set_seed(cfg)

    # Logdirs
    logdir_dict = set_logdir(cfg)

    # Dataset
    dataset_dict = set_datasets(cfg)

    # Data property
    aabb = dataset_dict["train_dataset"].scene_bbox.to(device)
    ndc_ray = dataset_dict["train_dataset"].ndc_ray
    white_bg = dataset_dict["train_dataset"].white_bg
    near_far = dataset_dict["train_dataset"].near_far

    # Model
    model_dict = create_model(cfg, device, aabb=aabb, near_far=near_far); model=model_dict["model"]
    trainer_render_dict = create_trainer_and_render(cfg, model, dataset_dict, logdir_dict["tb_writer"], logdir_dict["logdir_train_root"], device,
                                        reso_cur=model_dict["reso_cur"] if model_dict["reso_cur"] is not None else None)

    #####################
    # TRAINER OR RENDER #
    #####################
    if cfg.render_only and (cfg.render_test or cfg.render_train or cfg.render_path or cfg.render_val or cfg.render_disentangled or cfg.vis_planes or cfg.render_mesh):
        # model
        model.eval()
        
        # Render
        ndc_ray = dataset_dict["test_dataset"].ndc_ray
        white_bg = dataset_dict["test_dataset"].white_bg

        # set render/render_trajectory_fn
        render = trainer_render_dict["render_fn"]; render_trajectory = trainer_render_dict["render_trajectory_fn"]
        render_disentangled = trainer_render_dict["render_disentangled_fn"]; vis_planes = trainer_render_dict["vis_planes_fn"]
        render_mesh = trainer_render_dict["render_mesh_fn"]
        
        # Render 
        if cfg.render_test:
            ##############
            # NUM PARAMS #
            ##############
            if hasattr(model, "get_num_params"): 
                num_params = model.get_num_params()
            else: 
                num_params = 0
                grad_vars = model.get_optparam_groups(cfg.optim)
                
                for p in grad_vars:
                    if isinstance(p["params"], torch.nn.ParameterList):
                        for pp in p["params"]:
                            num_params += pp.numel()
                    else:
                        num_params += sum(pr.numel() for pr in p["params"])
            
            with open(f'{logdir_dict["logdir_render_test"]}/num_params.txt', 'w') as file: 
                file.write(f'num_params : {num_params}\n')
            
            render(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_test"],
                prefix="test", N_vis=-1, chunk=cfg.model.chunk, n_samples=cfg.model.n_samples if (cfg.model.n_samples < 1000000) and (cfg.model.n_samples > 0) else -1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            
        if cfg.render_val:
            render(dataset_dict["val_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_val"],
                prefix="test", N_vis=-1, chunk=cfg.model.chunk, n_samples=cfg.model.n_samples if (cfg.model.n_samples < 1000000) and (cfg.model.n_samples > 0) else -1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
                
        if cfg.render_train:
            render(dataset_dict["train_render_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_train"],
                prefix="test", N_vis=-1, chunk=cfg.model.chunk, n_samples=cfg.model.n_samples if (cfg.model.n_samples < 1000000) and (cfg.model.n_samples > 0) else -1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)

        if cfg.render_path:
            render_trajectory(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_trajectory"],
                prefix="test", N_vis=-1, chunk=cfg.model.chunk, n_samples=cfg.model.n_samples if (cfg.model.n_samples < 1000000) and (cfg.model.n_samples > 0) else -1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
        
        if cfg.render_mesh:
            render_mesh(model,
                cfg=cfg, savePath=logdir_dict["logdir_render_mesh"],
                prefix="test", device=device)
            
        if cfg.vis_planes:
            vis_planes(model,
                cfg=cfg, savePath=logdir_dict["logdir_render_test"],
                prefix="test", device=device)
                
        # only valid for TensorRefine_disentangled    
        if cfg.render_disentangled and (render_disentangled is not None):
            assert render_disentangled is not None, "render_disentangled_fn is not implemented"
            render_disentangled(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_disentangled"],
                prefix="disentangled", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)

    else:
        # Initialize render/render_trajectory_fn
        render = trainer_render_dict["render_fn"]; render_trajectory = trainer_render_dict["render_trajectory_fn"]
        render_disentangled = trainer_render_dict["render_disentangled_fn"]; vis_planes = trainer_render_dict["vis_planes_fn"]

        # Initialize trainer
        trainer = trainer_render_dict["trainer"]

        # Execute training
        model.train()
        
        # Wall Clock Time
        trainer.train()
        
        # Save model
        torch.save(model, f"{logdir_dict['logdir']}/{cfg.expname}.th")
        
        # eval
        model.eval()

        # Render 
        if cfg.render_test:
            # num_params
            if hasattr(model, "get_num_params"): 
                num_params = model.get_num_params()
            else: 
                num_params = 0
                grad_vars = model.get_optparam_groups(cfg.optim)
                
                for p in grad_vars:
                    if isinstance(p["params"], torch.nn.ParameterList):
                        for pp in p["params"]:
                            num_params += pp.numel()
                    else:
                        num_params += sum(pr.numel() for pr in p["params"])
                
                
            with open(f'{logdir_dict["logdir_render_test"]}/num_params.txt', 'w') as file: 
                file.write(f'num_params : {num_params}\n')
            
            render(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_test"],
                prefix="test", N_vis=-1, chunk=cfg.model.chunk, n_samples=cfg.model.n_samples if (cfg.model.n_samples < 1000000) and (cfg.model.n_samples > 0) else -1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            
        if cfg.render_val:
            render(dataset_dict["val_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_val"],
                prefix="test", N_vis=-1, chunk=cfg.model.chunk, n_samples=cfg.model.n_samples if (cfg.model.n_samples < 1000000) and (cfg.model.n_samples > 0) else -1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            
        if cfg.render_train:
            render(dataset_dict["train_render_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_train"],
                prefix="test", N_vis=-1, chunk=cfg.model.chunk, n_samples=cfg.model.n_samples if (cfg.model.n_samples < 1000000) and (cfg.model.n_samples > 0) else -1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)

        if cfg.render_path:
            render_trajectory(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_trajectory"],
                prefix="test", N_vis=-1, chunk=cfg.model.chunk, n_samples=cfg.model.n_samples if (cfg.model.n_samples < 1000000) and (cfg.model.n_samples > 0) else -1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)

        if cfg.vis_planes:
            vis_planes(model,
                cfg=cfg, savePath=logdir_dict["logdir_render_test"],
                prefix="test", device=device)
            
        # only valid for TensorRefine_disentangled
        if cfg.render_disentangled and (render_disentangled is not None):
            assert render_disentangled is not None, "render_disentangled_fn is not implemented"
            render_disentangled(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_disentangled"],
                prefix="test", N_vis=-1, chunk=cfg.model.chunk, n_samples=cfg.model.n_samples if (cfg.model.n_samples < 1000000) and (cfg.model.n_samples > 0) else -1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)