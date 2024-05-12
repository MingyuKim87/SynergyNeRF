import torch
from easydict import EasyDict as edict

from .nerf_dataset import NerfDataset
from .tankstemple import TanksTempleDataset
from .neural_3D_dataset_NDC import Neural3D_NDC_Dataset




def get_train_dataset(cfg, is_stack=False, device=torch.device("cuda")):
    
    if cfg.data.dataset_name == "nerf":
        train_dataset = NerfDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            sparse_option=cfg.data.sparse_option
        )
    elif cfg.data.dataset_name == "neural3D_NDC":
        train_dataset = Neural3D_NDC_Dataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )
    elif cfg.data.dataset_name == "TanksAndTemple":
        train_dataset = TanksTempleDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
        )
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset

def get_val_dataset(cfg, is_stack=True, device=torch.device("cuda")):
    if cfg.data.dataset_name == "nerf":
        val_dataset = NerfDataset(
            cfg.data.datadir,
            "val",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=-1,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
        )
    elif cfg.data.dataset_name == "neural3D_NDC":
        val_dataset = Neural3D_NDC_Dataset(
            cfg.data.datadir,
            "val",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )
    elif cfg.data.dataset_name == "TanksAndTemple":
        val_dataset = TanksTempleDataset(
            cfg.data.datadir,
            "val",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=-1,
        )
    else:
        raise NotImplementedError("No such dataset")
    return val_dataset


def get_test_dataset(cfg, is_stack=True, device=torch.device("cuda")):
    if cfg.data.dataset_name == "nerf":
        test_dataset = NerfDataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=-1,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
        )
    elif cfg.data.dataset_name == "neural3D_NDC":
        test_dataset = Neural3D_NDC_Dataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )
    elif cfg.data.dataset_name == "TanksAndTemple":
        test_dataset = TanksTempleDataset(
            cfg.data.datadir,
            "val",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=-1,
        )
    
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset


# move tensors to device in-place
def move_to_device(X, device):
    if isinstance(X,dict):
        for k,v in X.items():
            X[k] = move_to_device(v,device)
    elif isinstance(X,list):
        for i,e in enumerate(X):
            X[i] = move_to_device(e,device)
    elif isinstance(X,tuple) and hasattr(X,"_fields"): # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd,device)
        return type(X)(**dd)
    elif isinstance(X,torch.Tensor):
        return X.to(device=device)
    return X