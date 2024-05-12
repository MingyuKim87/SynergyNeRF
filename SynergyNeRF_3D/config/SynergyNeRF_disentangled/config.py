from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class System_Config:
    seed: int = 20220401
    basedir: str = "./results/SynergyNeRF_disentangled"
    ckpt: Optional[str] = None
    progress_refresh_rate: int = 10
    vis_every: int = 2000
    add_timestamp: bool = True


@dataclass
class Model_Config:
    model_name: str = "SynergyNeRF_disentangled" 
    N_voxel_init: int = 16 * 16 * 16  # initial voxel number
    N_voxel_final: int = 200 * 200 * 200  # final voxel number
    step_ratio: float = 0.5
    nonsquare_voxel: bool = True  # if yes, voxel numbers along each axis depend on scene length along each axis
    normalize_type: str = "normal"
    upsample_list: List[int] = field(default_factory=lambda: [3000, 6000, 9000])
    update_alpha_grid_mask_list: List[int] = field(
        default_factory=lambda: [4000, 8000, 10000]
    )

    # Plane Initialization
    density_n_comp: List[int] = field(default_factory=lambda: [48, 48, 48])
    density_dim: int = 1
    app_dim: int = 27
    density_mode: str = "Grid_Coord_Mixing_MLP"  # choose from "plain", "Grid_Coord_Mixing_MLP"
    app_mode: str = "general_MLP"
    init_scale: float = 0.1
    init_shift: float = 0.0

    # Fusion Methods
    fusion_one: str = "multiply" # only support multiply
    fusion_two: str = "concat" # choose from "sum" and "concat"

    # Density Feature Settings
    fea2dense_act: str = "softplus"
    density_shift: float = -10.0
    distance_scale: float = 25.0

    # Density Regressor MLP settings
    density_pos_pe: int = 0
    density_view_pe: int = -1
    density_fea_pe: int = -1 
    density_feature_dim: int = 256
    density_n_layers: int = 2

    # Appearance Regressor MLP settings
    app_pos_pe: int = -1
    app_view_pe: int = 2
    app_fea_pe: int = -1
    app_feature_dim: int = 256
    app_n_layers: int = 2

    # Alpha mask settings
    alpha_mask_thres: float = 0.001
    ray_march_weight_thres: float = 0.00001

    # Curriculum learning
    curriculum_num_modes : Optional[int] = 3
    curriculum_num_features : Optional[int] = 48
    curriculum_start : Optional[float] = 0.0
    curriculum_end : Optional[float] = 0.5

    # Reg
    random_background: bool = False
    depth_loss: bool = False
    depth_loss_weight: float = 1.0
    dist_loss: bool = False
    dist_loss_weight: float = 0.01

    plane_smooth_type : str = "LM" # choose from "TV" and "LM"
    TV_weight_density: float = 0.0001
    L1_weight_density: float = 8e-5
    L1_weight_density_rest: float = 4e-5
    

    # Sampling
    align_corners: bool = True
    # There are two types of upsampling: aligned and unaligned.
    # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
    # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
    # using "unaligned" upsampling will essentially double the grid sizes at each time, ignoring N_voxel_final.
    upsampling_type: str = "aligned"  # choose from "aligned", "unaligned".
    n_samples: int = 1000000

    # for test 
    chunk: int = 3500


@dataclass
class Data_Config:
    datadir: str = "./data/nerf_synthetic"
    scene: str = "lego"
    dataset_name: str = "nerf"  # choose from "dnerf", "neural3D_NDC"
    sparse_option: Optional[str] = None # choose from "None", "DietNeRF", "HALO"
    downsample: float = 1.0
    cal_fine_bbox: bool = False
    N_vis: int = -1
    time_scale: float = 1.0
    scene_bbox_min: List[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    scene_bbox_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    N_random_pose: int = 1000
    pose_radius_scale: float = -1.0
    # for dnerf

    # for neural3D_NDC
    nv3d_ndc_bd_factor: float = 0.75
    nv3d_ndc_eval_step: int = 1
    nv3d_ndc_eval_index: int = 0
    nv3d_ndc_sphere_scale: float = 1.0

    # Hierachical Sampling for Neural3D_NDC
    stage_1_iteration: int = 300000
    stage_2_iteration: int = 250000
    stage_3_iteration: int = 100000
    key_f_num: int = 10
    stage_1_gamma: float = 0.001
    stage_2_gamma: float = 0.02
    stage_3_alpha: float = 0.1

    datasampler_type: str = "rays"  # choose from "rays", "images", "hierach"


@dataclass
class Optim_Config:
    # Learning Rate
    lr_density_grid: float = 0.02
    lr_app_grid: float = 0.02
    lr_density_nn: float = 0.001
    lr_app_nn: float = 0.001

    # Optimizer, Adam deault
    beta1: float = 0.9
    beta2: float = 0.99
    lr_decay_type: str = "exp"  # choose from "exp" or "cosine" or "linear"
    lr_decay_target_ratio: float = 0.1
    lr_decay_step: int = -1
    lr_upsample_reset: bool = True

    batch_size: int = 4096
    n_iters: int = 30000


@dataclass
class Config:
    config: Optional[str] = None
    expname: Optional[str] = "default"

    render_only: bool = False
    render_train: bool = False
    render_val: bool = False
    render_test: bool = True
    render_path: bool = False
    render_disentangled : bool = False # only valid for TensorRefine_disentangled
    render_mesh: bool = False
    vis_planes : bool = False

    systems: System_Config = System_Config()
    model: Model_Config = Model_Config()
    data: Data_Config = Data_Config()
    optim: Optim_Config = Optim_Config()
