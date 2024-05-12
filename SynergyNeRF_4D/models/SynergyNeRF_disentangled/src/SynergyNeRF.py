import torch
from torch.nn import functional as F

from .SynergyNeRF_Base import SynergyNeRF_Base


class SynergyNeRF(SynergyNeRF_Base):
    """
    A general version of SynergyNeRF, which supports different fusion methods and feature regressor methods.
    """

    def __init__(self, aabb, grid_size, device, time_grid, near_far, **kargs):
        super().__init__(aabb, grid_size, device, time_grid, near_far, **kargs)

    def init_planes(self, res, device):
        """
        Initialize the planes. density_plane is the spatial plane while density_line_time is the spatial-temporal plane.
        """
        self.density_plane, self.density_line_time = self.init_one_hexplane(
            self.density_n_comp, self.grid_size, device
        )
        
        if (self.fusion_two != "concat"):  
            # if fusion_two is not concat, then we need dimensions from each paired planes are the same.
            assert self.density_n_comp[0] == self.density_n_comp[1]
            assert self.density_n_comp[0] == self.density_n_comp[2]

    def init_one_hexplane(self, n_component, grid_size, device):
        plane_coef, line_time_coef = [], []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef.append(
                torch.nn.Parameter(
                    self.init_shift + 
                    self.init_scale * torch.randn((
                        1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0]))
                    )
            )
            line_time_coef.append(
                torch.nn.Parameter(
                    self.init_shift + 
                    self.init_scale * torch.randn((
                        1, n_component[i], grid_size[vec_id], self.time_grid))
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), \
                torch.nn.ParameterList(line_time_coef).to(device)

    def get_optparam_groups(self, cfg, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.density_line_time,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            }
        ]

        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]

        if isinstance(self.app_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]

        return grad_vars

    def compute_tensorfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor, progress: float
    ) -> torch.Tensor:
        """
        Compuate the density features of sampled points from density SynergyNeRF.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            density: (N) density of sampled points.
        """
        ################################################
        # PREPARATION OF COORDINATES FOR INTERPOLATION #
        ################################################
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        ###########################
        # CALCULATE GRID FEATURES #
        ###########################
        plane_feat, line_time_feat = [], []

        # Extract features from six feature planes.
        for idx_plane in range(len(self.density_plane)):
            # Spatial Plane Feature: Grid sampling on density plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.density_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on density line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.density_line_time[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_feat, line_time_feat = torch.stack(plane_feat, dim=0), torch.stack(
            line_time_feat, dim=0
        )

        # Fusion One
        if self.fusion_one == "multiply":
            grid_feature = plane_feat * line_time_feat
        else:
            raise NotImplementedError("[Density Feature] Invalid fusion_1 type")

        # Fusion Two
        if self.fusion_two == "sum":
            grid_feature = torch.sum(grid_feature, dim=0)
        elif self.fusion_two == "concat":
            grid_feature = grid_feature.view(-1, grid_feature.shape[-1])
        else:
            raise NotImplementedError("[Density Feature] Invalid fusion_2 type")

        #######################################################
        # MIXING LOWFREQ COORDINATE FEATURE WITH GRID_FEATURE #
        #######################################################
        # curriculum weighting
        grid_feature = grid_feature.T
        grid_feature = self.curriculum_weighting_grid(grid_feature, progress)

        return grid_feature

    def TV_loss_density(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.density_plane)):
            total = (
                total + reg(self.density_plane[idx]) + reg2(self.density_line_time[idx])
            )
        return total

    def TV_loss_app(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) + reg2(self.app_line_time[idx])
        return total

    def L1_loss_density(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
                + torch.mean(torch.abs(self.density_line_time[idx]))
            )
        return total

    def L1_loss_app(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.app_plane[idx]))
                + torch.mean(torch.abs(self.app_line_time[idx]))
            )
        return total

    @torch.no_grad()
    def up_sampling_planes(self, plane_coef, line_time_coef, res_target, time_grid):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
            line_time_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_time_coef[i].data,
                    size=(res_target[vec_id], time_grid),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )

        return plane_coef, line_time_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        self.density_plane, self.density_line_time = self.up_sampling_planes(
            self.density_plane, self.density_line_time, res_target, time_grid
        )

        self.update_step_size(res_target)
        print(f"upsamping to {res_target}")
