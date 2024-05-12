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
    trainer_render_dict = create_trainer_and_render(cfg, model, dataset_dict["train_dataset"], dataset_dict["val_dataset"],
                                        logdir_dict["tb_writer"], logdir_dict["logdir_train_root"], device,
                                        reso_cur=model_dict["reso_cur"] if model_dict["reso_cur"] is not None else None)

    #####################
    # TRAINER OR RENDER #
    #####################
    if cfg.render_only and (cfg.render_test or cfg.render_path or cfg.render_val or cfg.render_disentangled):
        # model
        model.eval()
        
        # Render
        ndc_ray = dataset_dict["test_dataset"].ndc_ray
        white_bg = dataset_dict["test_dataset"].white_bg

        # set render/render_trajectory_fn
        render = trainer_render_dict["render_fn"]; render_trajectory = trainer_render_dict["render_trajectory_fn"]
        render_disentangled = trainer_render_dict["render_disentangled_fn"]
        vis_planes = trainer_render_dict["vis_planes_fn"]
        
        # Render 
        if cfg.render_test:
            render(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_test"],
                prefix="test", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            
        if cfg.render_val:
            render(dataset_dict["val_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_val"],
                prefix="val", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            
        if cfg.render_train:
            render(dataset_dict["train_render_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_train"],
                prefix="train", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)

        if cfg.render_path:
            render_trajectory(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_trajectory"],
                prefix="trajectory", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
                
        if cfg.vis_planes:
            vis_planes(model,
                cfg=cfg, savePath=logdir_dict["logdir_render_test"],
                prefix="test", device=device)
            
        if cfg.render_disentangled and (render_disentangled is not None):
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
        trainer.train()

        # Save model
        torch.save(model, f"{logdir_dict['logdir']}/{cfg.expname}.th")

        # eval
        model.eval()

        # Render 
        if cfg.render_test:
            render(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_test"],
                prefix="test", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            
        if cfg.render_val:
            render(dataset_dict["val_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_val"],
                prefix="val", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            
        if cfg.render_train:
            render(dataset_dict["train_render_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_train"],
                prefix="train", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)

        if cfg.render_path:
            render_trajectory(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_trajectory"],
                prefix="trajectory", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            
        if cfg.vis_planes:
            vis_planes(model,
                cfg=cfg, savePath=logdir_dict["logdir_render_test"],
                prefix="test", device=device)
            
        if cfg.render_disentangled and (render_disentangled is not None):
            assert render_disentangled is not None, "render_disentangled_fn is not implemented"
            render_disentangled(dataset_dict["test_dataset"], model,
                cfg=cfg, savePath=logdir_dict["logdir_render_disentangled"],
                prefix="disentangled", N_vis=-1, n_samples=-1,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device)