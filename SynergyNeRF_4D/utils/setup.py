import shutil
import datetime
import os
import random
import numpy as np
import torch

from config.HexPlane.config import Config
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_test_dataset, get_train_dataset, get_val_dataset

def set_config():
    cli_cfg = OmegaConf.from_cli() # command line
    yaml_path = cli_cfg.get("config", None) # yaml
    yaml_cfg = OmegaConf.load(yaml_path)

    # Default config
    if "Hex" in yaml_cfg.model.model_name:
        from config.HexPlane.config import Config
        base_cfg = OmegaConf.structured(Config())
    elif "SynergyNeRF" in yaml_cfg.model.model_name:
        if not len(yaml_cfg.model.model_name.split("_")) > 1:
            from config.SynergyNeRF.config import Config
            base_cfg = OmegaConf.structured(Config())
        elif "disentangled" in yaml_cfg.model.model_name:
            from config.SynergyNeRF_disentangled.config import Config
            base_cfg = OmegaConf.structured(Config())
        else:
            NotImplementedError     
    else:
        NotImplementedError 
    
    # merge configs
    cfg = OmegaConf.merge(base_cfg, yaml_cfg, cli_cfg)  

    # data dir / expname
    cfg.data.datadir = os.path.join(cfg.data.datadir, cfg.data.scene)
    cfg.expname = f"{cfg.model.model_name}_{cfg.data.scene}"

    # curriculum learning
    if ("SynergyNeRF" in yaml_cfg.model.model_name) and cfg.model.curriculum_start:
        cfg.model.curriculum_num_modes = len(cfg.model.density_n_comp) \
            if cfg.model.fusion_two == "concat" else 1
        cfg.model.curriculum_num_features = cfg.model.density_n_comp[0]

    return cfg

def set_seed(cfg):
    # Fix Random Seed for Reproducibility.
    random.seed(cfg.systems.seed)
    np.random.seed(cfg.systems.seed)
    torch.manual_seed(cfg.systems.seed)
    torch.cuda.manual_seed(cfg.systems.seed)

def set_logdir(cfg):
    # Initialize log_dirs
    if cfg.systems.ckpt is not None:
        log_rootdir = os.path.dirname(cfg.systems.ckpt)
    else:
        log_rootdir = f'{cfg.systems.basedir}/{cfg.expname}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        
        # make train dirs
        os.makedirs(log_rootdir, exist_ok=True); train_path = f"{log_rootdir}/train_vis"
        os.makedirs(train_path, exist_ok=True)
        summary_writer = SummaryWriter(os.path.join(log_rootdir, "tb_log"))
        cfg_file = os.path.join(f"{log_rootdir}", "cfg.yaml")
        with open(cfg_file, "w") as f:
            OmegaConf.save(config=cfg, f=f)
    
    # Render Test
    render_root_path = f"{log_rootdir}/render"; os.makedirs(render_root_path, exist_ok=True)
    render_train_path = f"{render_root_path}/train_all"
    render_val_path = f"{render_root_path}/val_all"
    render_test_path = f"{render_root_path}/test_all"
    render_trajectory_path = f"{render_root_path}/trajectory_all"
    render_disentangled_path = f"{render_root_path}/disentangled_all" 
    
    # make dirs
    if cfg.render_train: os.makedirs(render_train_path, exist_ok=True) 
    if cfg.render_val: os.makedirs(render_val_path, exist_ok=True)
    if cfg.render_test: os.makedirs(render_test_path, exist_ok=True) 
    if cfg.render_path: os.makedirs(render_trajectory_path, exist_ok=True)
    if cfg.render_disentangled: os.makedirs(render_trajectory_path, exist_ok=True)

    # save cofig
    cfg_file = os.path.join(f"{render_root_path}", "cfg_render.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    
    return {
        "logdir" : log_rootdir,
        "logdir_render_root" : render_root_path,
        "logdir_render_train" : render_train_path if cfg.render_train else None,
        "logdir_render_val" : render_val_path if cfg.render_val else None,
        "logdir_render_test" : render_test_path if cfg.render_test else None,
        "logdir_render_trajectory" : render_trajectory_path if cfg.render_path else None,
        "logdir_render_disentangled" : render_disentangled_path if cfg.render_disentangled else None,
        "logdir_train_root" : train_path if cfg.systems.ckpt is None else None,
        "tb_writer" : summary_writer if cfg.systems.ckpt is None else None
    }

    
def set_datasets(cfg):
    # train dataset
    train_dataset = get_train_dataset(cfg, is_stack=False)
    val_dataset = get_val_dataset(cfg, is_stack=True)
    test_dataset = get_test_dataset(cfg, is_stack=True)
    
    if cfg.render_train: train_render_dataset = get_train_dataset(cfg, is_stack=True)

    return {
        "test_dataset" : test_dataset,
        "val_dataset" : val_dataset,
        "train_dataset" : train_dataset,
        "train_render_dataset" : train_render_dataset if cfg.render_train else None
    }