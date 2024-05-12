import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn
from torch.nn import functional as F

from .curriculum_weighting import curriculum_weighting
from .mlp import General_MLP, Grid_Coord_Mixing_MLP
from .mlp import TCNN_General_MLP, TCNN_Grid_Coord_Mixing_MLP 
from .sh import eval_sh_bases

def raw2alpha(sigma: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    alpha = 1.0 - torch.exp(-sigma * dist)

    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )

    weights = alpha * T[:, :-1]  # [n_rays, n_samples]
    return alpha, weights, T[:, -1:]

def SHRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb

def SH_featrue(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    sh = eval_sh_bases(3, viewdirs)
    return sh

def RGBRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    rgb = features
    return rgb

def DensityRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    density = features
    return density


class alpha_grid_mask(torch.nn.Module):
    def __init__(
        self, device: torch.device, aabb: torch.Tensor, empty_volume: torch.Tensor
    ):
        super().__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabb_size = self.aabb[1] - self.aabb[0]
        self.inv_grid_size = 1.0 / self.aabb_size * 2
        self.empty_volume = empty_volume.view(1, 1, *empty_volume.shape[-3:])
        self.grid_size = torch.LongTensor(
            [empty_volume.shape[-1], empty_volume.shape[-2], empty_volume.shape[-3]]
        ).to(self.device)

    def sample_empty(self, xyz_sampled):
        empty_vals = F.grid_sample(
            self.empty_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True
        ).view(-1)
        return empty_vals


class SynergyNeRF_Base(torch.nn.Module):
    """
    SynergyNeRF Base Class.
    """

    def __init__(
        self,
        aabb: torch.Tensor,
        grid_size: List[int],
        near_far: List[float],
        device: torch.device,
        density_n_comp: Union[int, List[int]] = 48,
        density_dim: int = 1,
        app_dim: int = 27,
        density_mode: str = "plain",
        app_mode: str = "general_MLP",
        alpha_mask: Optional[alpha_grid_mask] = None,
        fusion_one: str = "multiply",
        fusion_two: str = "concat",
        fea2dense_act: str = "softplus",
        init_scale: float = 0.1,
        init_shift: float = 0.0,
        normalize_type: str = "normal",
        **kwargs,
    ):
        super().__init__()
        self.fp16 = kwargs.get("fp16", False)
        self.aabb = aabb
        self.device = device
        self.near_far = near_far
        self.near_far_org = near_far
        self.step_ratio = kwargs.get("step_ratio", 2.0)
        self.update_step_size(grid_size)

        # Density and Appearance SynergyNeRF components numbers and value regression mode.
        self.density_n_comp = density_n_comp
        self.density_dim = density_dim
        self.app_dim = app_dim
        self.align_corners = kwargs.get("align_corners", True)  # align_corners for grid_sample

        # Tensor weights initialization: scale and shift for uniform distribution.
        self.init_scale = init_scale
        self.init_shift = init_shift

        # Tensor feature fusion mode.
        self.fusion_one = fusion_one
        self.fusion_two = fusion_two

        # Plane Index
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]

        # Coordinate normalization type.
        self.normalize_type = normalize_type

        # Plane initialization.
        self.init_planes(grid_size[0], device)

        # Density calculation settings.
        self.fea2dense_act = fea2dense_act  # feature to density activation function
        self.density_shift = kwargs.get("density_shift", -10.0)  # density shift for density activation function.
        self.distance_scale = kwargs.get("distance_scale", 25.0)  # distance scale for density activation function.
        self.density_mode = density_mode
        
        self.density_pos_pe = kwargs.get("density_pos_pe", -1)
        self.density_view_pe = kwargs.get("density_view_pe", -1)
        self.density_fea_pe = kwargs.get("density_fea_pe", 6)
        self.density_feature_dim = kwargs.get("density_feature_dim", 128)
        self.density_n_layers = kwargs.get("density_n_layers", 3)
        self.init_density_func(
            self.density_mode,
            self.density_pos_pe,
            self.density_feature_dim,
            self.density_n_layers,
            self.device)

        # Appearance calculation settings.
        self.app_mode = app_mode
        self.app_pos_pe = kwargs.get("app_pos_pe", -1)
        self.app_view_pe = kwargs.get("app_view_pe", 6)
        self.app_fea_pe = kwargs.get("app_fea_pe", 6)
        self.app_feature_dim = kwargs.get("app_feature_dim", 128)
        self.app_n_layers = kwargs.get("app_n_layers", 3)
        self.init_app_func(
            app_mode,
            self.app_pos_pe,
            self.app_view_pe,
            self.app_fea_pe,
            self.app_feature_dim,
            self.app_n_layers,
            device)

        # Density SynergyNeRF mask and other acceleration tricks.
        self.alpha_mask = alpha_mask
        self.alpha_mask_thres = kwargs.get("alpha_mask_thres", 0.001)  # density threshold for emptiness mask
        self.ray_march_weight_thres = kwargs.get("rayMarch_weight_thres", 0.0001)  # density threshold for rendering colors.

        # Curriculum learning
        curriculum_num_modes = kwargs.get("curriculum_num_modes", None)
        curriculum_num_features = kwargs.get("curriculum_num_features", None)
        curriculum_start = kwargs.get("curriculum_start", None)
        curriculum_end = kwargs.get("curriculum_end", None)

        self.curriculum_weighting_grid = curriculum_weighting(
            num_modes = curriculum_num_modes,
            num_features = curriculum_num_features,
            start = curriculum_start, end = curriculum_end)

        # Regulartization settings.
        self.random_background = kwargs.get("random_background", False)
        self.depth_loss = kwargs.get("depth_loss", False)

    def init_density_func(self, density_mode, pos_pe, feature_dim, n_layers, device):
        """Initialize density regression function."""
        if (density_mode == "plain"):  # Use extracted features directly from SynergyNeRF as density.
            assert self.density_dim == 1  # Assert the extracted features are scalers.
            self.density_regressor = DensityRender    
        elif (density_mode == "Grid_Coord_Mixing_MLP"):
            self.density_regressor = Grid_Coord_Mixing_MLP(
                output_dim=1+self.app_dim,
                input_pos_dim=3,
                input_feat_dim=sum(self.density_n_comp) if self.fusion_two == "concat" else self.density_n_comp[0],
                pos_pe=pos_pe, feature_dim=feature_dim, n_layers=n_layers, zero_init=True, fp16=self.fp16).to(device)
        elif (density_mode == "TCNN_Grid_Coord_Mixing_MLP"):
            self.density_regressor = TCNN_Grid_Coord_Mixing_MLP(
                output_dim=1+self.app_dim,
                input_pos_dim=3,
                input_feat_dim=sum(self.density_n_comp) if self.fusion_two == "concat" else self.density_n_comp[0],
                pos_pe=pos_pe, feature_dim=feature_dim, n_layers=n_layers, zero_init=True, fp16=self.fp16).to(device)
        else:
            raise NotImplementedError("Invalid Density Regression Mode")
        
        print("[Model] Density regressor:")
        print(self.density_regressor)
        print("*"*10, f"pos_pe : {pos_pe}", "*"*10)

    def init_app_func(self, app_mode, pos_pe, view_pe, fea_pe, feature_dim, n_layers, device):
        """Initialize appearance regression function."""
        if app_mode == "SH":  # Use Spherical Harmonics SH to render appearance.
            self.app_regressor = SHRender

        elif app_mode == "RGB":  # Use RGB to render appearance.
            assert self.app_dim == 3
            self.app_regressor = RGBRender

        elif app_mode == "general_MLP":  # Use general MLP to render appearance.
            SH_feature_dim = 16 # degree : 3
            self.app_regressor = General_MLP(SH_feature_dim+self.app_dim, 3, 
                                             fea_pe, pos_pe, view_pe, 
                                             feature_dim, n_layers, 
                                             use_sigmoid=True, zero_init=True, fp16=self.fp16).to(device)
        elif app_mode == "tcnn_general_MLP":  # Use general MLP to render appearance.
            SH_feature_dim = 16 # degree : 3
            self.app_regressor =  TCNN_General_MLP(SH_feature_dim+self.app_dim, 3, 
                                             fea_pe, pos_pe, view_pe, 
                                             feature_dim, n_layers, 
                                             use_sigmoid=True, zero_init=True, fp16=self.fp16).to(device)
        else:
            raise NotImplementedError("Invalid App Regression Mode")
        
        print("[Model] App regressor:")
        print(self.app_regressor)
        print("*"*10, f"pos_pe : {pos_pe} view_pe : {view_pe} fea_pe : {fea_pe}", "*"*10)

    def update_step_size(self, grid_size):
        self.aabb_size = self.aabb[1] - self.aabb[0]
        self.inv_aabb_size = 2.0 / self.aabb_size
        self.grid_size = torch.LongTensor(grid_size).to(self.device)
        self.units = self.aabb_size / (self.grid_size - 1)
        self.step_size = torch.mean(self.units) * self.step_ratio
        self.aabb_diag = torch.sqrt(torch.sum(torch.square(self.aabb_size)))
        self.n_samples = int((self.aabb_diag / self.step_size).item()) + 1
        
        print("*"*10, "Ray-Marching" , "*"*10)
        print(f"[Raymarching] AABB : {self.aabb.view(-1)}")
        print(f"[Raymarching] Grid Size : {grid_size}")
        print(f"[Raymarching] Raymarching step_size : {self.step_size}")
        print(f"[Raymarching] The number of samples for a ray: {self.n_samples}")
        print("*"*30)

    def init_planes(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_tensorfeature(self, xyz_sampled, progress):
        pass

    def compute_appfeature(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        """
        Normalize the sampled coordinates to [-1, 1] range.
        """
        if self.normalize_type == "normal":
            return (xyz_sampled - self.aabb[0]) * self.inv_aabb_size - 1

    def feature2density(self, density_features: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.fp16):
            dtype = torch.half if self.fp16 else torch.float
            if self.fea2dense_act == "softplus":
                return F.softplus(density_features + self.density_shift).to(dtype)
            elif self.fea2dense_act == "relu":
                return F.relu(density_features).to(dtype)
            else:
                raise NotImplementedError("No such activation function for density feature")

    def sample_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        is_train: bool = True,
        n_samples: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points along rays based on the given ray origin and direction.

        Args:
            rays_o: (B, 3) tensor, ray origin.
            rays_d: (B, 3) tensor, ray direction.
            is_train: bool, whether in training mode.
            n_samples: int, number of samples along each ray.

        Returns:
            rays_pts: (B, n_samples, 3) tensor, sampled points along each ray.
            interpx: (B, n_samples) tensor, sampled points' distance to ray origin.
            ~mask_outbbox: (B, n_samples) tensor, mask for points within bounding box.
        """
        n_samples = n_samples if n_samples > 0 else self.n_samples
        near, far = self.near_far
        interpx = torch.linspace(near, far, n_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / n_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )
        return rays_pts, interpx, ~mask_outbbox

    @torch.no_grad()
    def filtering_rays(
        self,
        all_rays: torch.Tensor,
        all_rgbs: torch.Tensor,
        all_depths: Optional[torch.Tensor] = None,
        n_samples: int = 256,
        chunk: int = 10240 * 5,
        bbox_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Filter out rays that are not within the bounding box.

        Args:
            all_rays: (n_rays, n_samples, 6) tensor, rays [rays_o, rays_d].
            all_rgbs: (n_rays, n_samples, 3) tensor, rgb values.
            all_depths: (n_rays, n_samples) tensor, depth values.
            n_samples: int, number of samples along each ray.

        Returns:
            all_rays: (n_rays, n_samples, 6) tensor, filtered rays [rays_o, rays_d].
            all_rgbs: (n_rays, n_samples, 3) tensor, filtered rgb values.
            all_depths: Optional, (n_rays, n_samples) tensor, filtered depth values.
        """
        print("="*10, "filtering rays", "="*10)
        tt = time.time()
        N = torch.tensor(all_rays.shape[:-1]).prod()
        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            # Filter based on bounding box.
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(
                    -1
                )  # clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(
                    -1
                )  # clamp(min=near, max=far)
                mask_inbbox = t_max > t_min
            # Filter based on emptiness mask.
            else:
                xyz_sampled, _, _ = self.sample_ray(
                    rays_o, rays_d, n_samples=n_samples, is_train=False
                )
                xyz_sampled = self.normalize_coord(xyz_sampled)
                mask_inbbox = (
                    self.alpha_mask.sample_empty(xyz_sampled).view(xyz_sampled.shape[:-1]
                    ) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f"[Filtering all rays in traning data] Ray filtering takes : {time.time()-tt} s.")
        print(f"[Filtering all rays in traning data] The ratio of dropped ray : {(torch.sum(mask_filtered) / N):.6f}")
              
        if all_depths is not None:
            return (
                all_rays[mask_filtered],
                all_rgbs[mask_filtered],
                all_depths[mask_filtered],
            )
        else:
            return (
                all_rays[mask_filtered],
                all_rgbs[mask_filtered],
                None,
            )

    def forward(
        self,
        rays_chunk: torch.Tensor,
        white_bg: bool = True,
        is_train: bool = False,
        ndc_ray: bool = False,
        n_samples: int = -1,
        progress: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SynergyNeRF.

        Args:
            rays_chunk: (B, 6) tensor, rays [rays_o, rays_d].
            white_bg: bool, whether to use white background.
            is_train: bool, whether in training mode.
            ndc_ray: bool, whether to use normalized device coordinates.
            n_samples: int, number of samples along each ray.

        Returns:
            rgb: (B, 3) tensor, rgb values.
            depth: (B, 1) tensor, depth values.
            alpha: (B, 1) tensor, accumulated weights.
            z_vals: (B, n_samples) tensor, z values.
        """
        # Prepare rays.
        with torch.cuda.amp.autocast(enabled=self.fp16):
            dtype = torch.half if self.fp16 else torch.float
            rays_chunk = rays_chunk.to(dtype)
            viewdirs = rays_chunk[:, 3:6]
            xyz_sampled, z_vals, ray_valid = self.sample_rays(
                rays_chunk[:, :3], viewdirs, is_train=is_train, n_samples=n_samples
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1
            )
            
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            
            if ndc_ray:
                dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm

            viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
            
            # Normalize coordinates.
            xyz_sampled = self.normalize_coord(xyz_sampled)

            # If emptiness mask is availabe, we first filter out rays with low opacities.
            if self.alpha_mask is not None:
                emptiness = self.alpha_mask.sample_empty(xyz_sampled[ray_valid])
                empty_mask = emptiness > 0
                ray_invalid = ~ray_valid
                ray_invalid[ray_valid] |= ~empty_mask
                ray_valid = ~ray_invalid

            # Initialize sigma and rgb values.
            dtype = torch.half if self.fp16 else torch.float
            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device, dtype=dtype)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device, dtype=dtype)
            app_features = torch.zeros((*xyz_sampled.shape[:2], self.app_dim), device=xyz_sampled.device, dtype=dtype)
            # Compute density feature and density if there are valid rays.
            if ray_valid.any():
                density_feature = self.compute_tensorfeature(xyz_sampled[ray_valid], progress)
                density_feature = self.density_regressor(
                    xyz_sampled[ray_valid], viewdirs[ray_valid], density_feature,
                )
                # Split
                app_features[ray_valid] = density_feature[..., 1:]; density_feature = density_feature[..., 0]
                
                # Calcuate sigma
                validsigma = self.feature2density(density_feature)
                sigma[ray_valid] = validsigma.view(-1)
            
            # alpha is the opacity, weight is the accumulated weight. bg_weight is the accumulated weight for last sampling point.
            alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)  

            # Compute appearance feature and rgb if there are valid rays (whose weight are above a threshold).
            app_mask = weight > self.ray_march_weight_thres

            # Update app_features using SH features
            SH_features = SH_featrue(xyz_sampled=None, viewdirs=(viewdirs + 1)/2, features=None)
            app_features = torch.cat((SH_features, app_features), dim=-1)

            # Make rgbs
            if app_mask.any():
                # Calculate rgbs
                valid_rgbs = self.app_regressor(xyz_sampled[app_mask], viewdirs[app_mask], app_features[app_mask])
                rgb[app_mask] = valid_rgbs
            
            acc_map = torch.sum(weight, -1)
            rgb_map = torch.sum(weight[..., None] * rgb, -2)

            # If white_bg or (is_train and torch.rand((1,))<0.5):
            if white_bg or not is_train:
                rgb_map = rgb_map + (1.0 - acc_map[..., None])
            else:
                rgb_map = rgb_map + (1.0 - acc_map[..., None]) * torch.rand(
                    size=(1, 3), device=rgb_map.device
                )
            rgb_map = rgb_map.clamp(0, 1)

            # Calculate depth.
            if self.depth_loss:
                depth_map = torch.sum(weight * z_vals, -1)
                depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]
            else:
                with torch.no_grad():
                    depth_map = torch.sum(weight * z_vals, -1)
                    depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]
        return rgb_map, depth_map, alpha, z_vals

    @torch.no_grad()
    def update_alpha_mask(self, grid_size=(200, 200, 200), progress=1.0):
        """
        Like TensoRF, we compute the emptiness voxel to store the opacities of the scene and skip computing the opacities of rays with low opacities.
        For SynergyNeRF, the emptiness voxel is the union of the density volumes of all the frames.

        This is the same idea as AlphaMask in TensoRF, while we rename it for better understanding.

        Note that we assume the voxel follows the current AABB dimensions, and we sample for normalized coordinate.
        """
        ks = 3 # kernel size for padding
        dtype = torch.half if self.fp16 else torch.float
        alpha_grid, dense_xyz = self.get_alpha_grid(grid_size, progress)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha_grid = alpha_grid.clamp(0, 1).transpose(0, 2).contiguous()[None, None]

        # Pooling
        alpha_grid = F.max_pool3d(alpha_grid, kernel_size=ks, padding=ks // 2, stride=1).view(grid_size[::-1])
        
        # Clamp
        alpha_grid[alpha_grid >= self.alpha_mask_thres] = 1
        alpha_grid[alpha_grid < self.alpha_mask_thres] = 0

        # Update
        self.alpha_mask = alpha_grid_mask(self.device, self.aabb, alpha_grid)

        return None

    @torch.no_grad()
    def get_alpha_grid(self, grid_size=None, progress=1.0):
        """
        For a 3D volume, we sample the opacity values of discrete space points and store them in a 3D volume.
        Note that we always assume the 3D volume is in the range of [min(aabb), max(aabb)] for each axis.
        """
        grid_size = self.grid_size if grid_size is None else grid_size
        dtype = torch.half if self.fp16 else torch.float
        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, grid_size[0]),
                torch.linspace(0, 1, grid_size[1]),
                torch.linspace(0, 1, grid_size[2]),
            ),
            -1,
        ).to(dtype).to(self.device)
        dense_xyz = samples * 2.0 - 1.0
        alpha_grid = torch.zeros_like(dense_xyz[..., 0])
        for i in range(grid_size[0]):
            alpha_grid[i] = self.compute_alpha_for_alpha_grid(
                dense_xyz[i].view(-1, 3).contiguous(), self.step_size, progress
            ).view((grid_size[1], grid_size[2]))
        return alpha_grid, dense_xyz

    def compute_alpha_for_alpha_grid(self, xyz_locs, length=1, progress=1.0):
        """
        Compute the emptiness of space points. Emptiness is the density.
        For each space point, we calcualte its densitis and calculate alpha (opacity).
        """
        dtype = torch.half if self.fp16 else torch.float
        if self.alpha_mask is not None:
            density_grid = self.alpha_mask.sample_empty(xyz_locs)
            density_mask = density_grid > 0
        else:
            density_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device, dtype=dtype)
        
        if density_mask.any():
            xyz_sampled = xyz_locs[density_mask]
            
            density_feature = self.compute_tensorfeature(xyz_sampled, progress)
            density_feature = self.density_regressor(xyz_sampled, xyz_sampled, density_feature)

            # Split
            sigma_feature = density_feature[..., 0]
            validsigma = self.feature2density(sigma_feature)
            sigma[density_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha
    
    @torch.no_grad()
    def get_tensorial_features(self):
        density_plane_clone_list = []; density_line_clone_list = []

        for i in range(len(self.density_plane)):
            density_plane_comp = np.squeeze(self.density_plane[i].clone().detach().cpu().numpy())
            density_line_comp = np.squeeze(self.density_line[i].clone().detach().cpu().numpy())

            density_plane_clone_list.append(density_plane_comp)
            density_line_clone_list.append(density_line_comp)
        
        
        return density_plane_clone_list, density_line_clone_list

    def get_optparam_groups(self, cfg, lr_scale=1.0):
        pass
    
    def get_num_params(self):
        pass