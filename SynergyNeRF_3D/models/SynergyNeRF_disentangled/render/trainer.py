import math
import sys

import time
import numpy as np
import torch
from tqdm.auto import tqdm

from .render import OctreeRender_trilinear_fast as renderer
from .render import render, render_disentangled
from .util.Reg import TVLoss, LMLoss, compute_dist_loss
from .util.Sampling import GM_Resi, cal_n_samples
from .util.util import N_to_reso


class SimpleSampler:
    """
    A sampler that samples a batch of ids randomly.
    """

    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


class Trainer:
    def __init__(
        self,
        model,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
    ):
        self.model = model
        self.cfg = cfg
        self.reso_cur = reso_cur
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.summary_writer = summary_writer
        self.logfolder = logfolder
        self.device = device

    # def get_lr_decay_factor(self, step):
    #     """
    #     Calculate the learning rate decay factor = current_lr / initial_lr.
    #     """
    #     if self.cfg.optim.lr_decay_step == -1:
    #         self.cfg.optim.lr_decay_step = self.cfg.optim.n_iters

    #     if self.cfg.optim.lr_decay_type == "exp":  # exponential decay
    #         lr_factor = self.cfg.optim.lr_decay_target_ratio ** (
    #             step / self.cfg.optim.lr_decay_step
    #         )
    #     elif self.cfg.optim.lr_decay_type == "linear":  # linear decay
    #         lr_factor = self.cfg.optim.lr_decay_target_ratio + (
    #             1 - self.cfg.optim.lr_decay_target_ratio
    #         ) * (1 - step / self.cfg.optim.lr_decay_step)
    #     elif self.cfg.optim.lr_decay_type == "cosine":  # consine decay
    #         lr_factor = self.cfg.optim.lr_decay_target_ratio + (
    #             1 - self.cfg.optim.lr_decay_target_ratio
    #         ) * 0.5 * (1 + math.cos(math.pi * step / self.cfg.optim.lr_decay_step))

    #     return lr_factor

    def get_voxel_upsample_list(self):
        """
        Precompute  spatial and temporal grid upsampling sizes.
        """
        upsample_list = self.cfg.model.upsample_list
        if (
            self.cfg.model.upsampling_type == "unaligned"
        ):  # logaritmic upsampling. See explation of "unaligned" in model/__init__.py.
            N_voxel_list = (
                torch.round(
                    torch.exp(
                        torch.linspace(
                            np.log(self.cfg.model.N_voxel_init),
                            np.log(self.cfg.model.N_voxel_final),
                            len(upsample_list) + 1,
                        )
                    )
                ).long()
            ).tolist()[1:]
        elif (
            self.cfg.model.upsampling_type == "aligned"
        ):  # aligned upsampling doesn't need precompute N_voxel_list.
            N_voxel_list = None
        
        self.N_voxel_list = N_voxel_list
        

    def sample_data(self, train_dataset, iteration):
        """
        Sample a batch of data from the dataset.
        """
        train_depth = None
        # sample rays: shuffle all the rays of training dataset and sampled a batch of rays from them.
        if self.cfg.data.datasampler_type == "rays":
            ray_idx = self.sampler.nextids()
            data = train_dataset[ray_idx]
            rays_train, rgb_train = (
                data["rays"],
                data["rgbs"].to(self.device),
            )
            if self.depth_data:
                train_depth = data["depths"].to(self.device)
        # sample images: randomly pick one image from the training dataset and sample a batch of rays from all the rays of the image.
        elif self.cfg.data.datasampler_type == "images":
            img_i = self.sampler.nextids()
            data = train_dataset[img_i]
            rays_train, rgb_train = (
                data["rays"],
                data["rgbs"].to(self.device).view(-1, 3)
            )
            select_inds = torch.randperm(rays_train.shape[0])[
                : self.cfg.optim.batch_size
            ]
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]

            if self.depth_data:
                train_depth = data["depths"].to(self.device).view(-1, 1)[select_inds]
        # hierarchical sampling from dyNeRF: hierachical sampling involves three stages of samplings.
        '''
        elif self.cfg.data.datasampler_type == "hierach": 
            # Stage 1: randomly sample a single image from an arbitrary camera.
            # And sample a batch of rays from all the rays of the image based on the difference of global median and local values.
            # Stage 1 only samples key-frames, which is the frame every self.cfg.data.key_f_num frames.
            if iteration <= self.cfg.data.stage_1_iteration:
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                index_i = np.random.choice(
                    train_dataset.all_rgbs.shape[1] // self.cfg.data.key_f_num
                )
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i * self.cfg.data.key_f_num]
                    .view(-1, 3)
                    .to(self.device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                
                # Calcualte the probability of sampling each ray based on the difference of global median and local values.
                probability = GM_Resi(
                    rgb_train, self.global_mean[cam_i], self.cfg.data.stage_1_gamma
                )
                select_inds = torch.multinomial(
                    probability, self.cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                
            elif (
                iteration
                <= self.cfg.data.stage_2_iteration + self.cfg.data.stage_1_iteration
            ):
                # Stage 2: basically the same as stage 1, but samples all the frames instead of only key-frames.
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                index_i = np.random.choice(train_dataset.all_rgbs.shape[1])
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(self.device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                
                probability = GM_Resi(
                    rgb_train, self.global_mean[cam_i], self.cfg.data.stage_2_gamma
                )
                select_inds = torch.multinomial(
                    probability, self.cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                
            else:
                # Stage 3: randomly sample one frame and sample a batch of rays from the sampled frame.
                # TO sample a batch of rays from this frame, we calcualate the value changes of rays compared to nearby timesteps, and sample based on the value changes.
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                N_time = train_dataset.all_rgbs.shape[1]
                # Sample two adjacent time steps within a range of 25 frames.
                index_i = np.random.choice(N_time)
                index_2 = np.random.choice(
                    min(N_time, index_i + 25) - max(index_i - 25, 0)
                ) + max(index_i - 25, 0)
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(self.device)
                )
                rgb_ref = (
                    train_dataset.all_rgbs[cam_i, index_2].view(-1, 3).to(self.device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[cam_i, index_i]
                # Calcualte the temporal difference between the two frames as sampling probability.
                probability = torch.clamp(
                    1 / 3 * torch.norm(rgb_train - rgb_ref, p=1, dim=-1),
                    min=self.cfg.data.stage_3_alpha,
                )
                select_inds = torch.multinomial(
                    probability, self.cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
        '''
        return rays_train, rgb_train, train_depth

    def init_sampler(self, train_dataset):
        """
        Initialize the sampler for the training dataset.
        """
        if self.cfg.data.datasampler_type == "rays":
            self.sampler = SimpleSampler(len(train_dataset), self.cfg.optim.batch_size)
        elif self.cfg.data.datasampler_type == "images":
            self.sampler = SimpleSampler(len(train_dataset), 1)
        elif self.cfg.data.datasampler_type == "hierach":
            self.global_mean = train_dataset.global_mean_rgb.to(self.device)

    def train(self):
        torch.cuda.empty_cache()

        # load the training and testing dataset and other settings.
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        model = self.model
        self.depth_data = test_dataset.depth_data
        summary_writer = self.summary_writer
        reso_cur = self.reso_cur

        ndc_ray = train_dataset.ndc_ray  # if the rays are in NDC
        white_bg = test_dataset.white_bg  # if the background is white

        # Calculate the number of samples for each ray based on the current resolution.
        n_samples = cal_n_samples(reso_cur, self.cfg.model.step_ratio, 
                                  max_value=self.cfg.model.n_samples)
        

        # Filter the rays based on the bbox
        if (self.cfg.data.datasampler_type == "rays") and (ndc_ray is False):
            allrays, allrgbs = (train_dataset.all_rays, train_dataset.all_rgbs)
            if self.depth_data:
                alldepths = train_dataset.all_depths
            else:
                alldepths = None

            allrays, allrgbs, alldepths = model.filtering_rays(
                allrays, allrgbs, alldepths, bbox_only=True)
            
            train_dataset.all_rays = allrays
            train_dataset.all_rgbs = allrgbs
            train_dataset.all_depths = alldepths

        # initialize the data sampler
        self.init_sampler(train_dataset)
        # precompute the voxel upsample list
        self.get_voxel_upsample_list()

        # Initialiaze TV loss on planse
        tvreg = TVLoss()  # TV loss on the spatial planes and lines

        # Initialiaze TV loss or LM loss on planse
            # TV loss on the spatial planes and lines
        if self.cfg.model.plane_smooth_type == "TV": tvreg = TVLoss()  
            # LM loss on the spatial planes    
        elif self.cfg.model.plane_smooth_type == "LM": tvreg = LMLoss()
        else: NotImplementedError  
        
            
        
        pbar = tqdm(
            range(self.cfg.optim.n_iters),
            miniters=self.cfg.systems.progress_refresh_rate,
            file=sys.stdout,
        )

        PSNRs, PSNRs_test = [], [0]
        torch.cuda.empty_cache()

        # Initialize the optimizer
        grad_vars = model.get_optparam_groups(self.cfg.optim)
        optimizer = torch.optim.Adam(
            grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.cfg.optim.lr_decay_target_ratio**(1/self.cfg.optim.n_iters))

        start_time = time.time()
        for iteration in pbar:
            # Sample dat
            rays_train, rgb_train, depth = self.sample_data(
                train_dataset, iteration
            )
            # Render the rgb values of rays
            rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
                rays_train,
                model,
                chunk=self.cfg.optim.batch_size,
                n_samples=n_samples,
                white_bg=white_bg,
                ndc_ray=ndc_ray,
                device=self.device,
                progress=iteration/self.cfg.optim.n_iters if iteration>=0 else 1.0,
                is_train=True,
            )

            # Calculate the loss
            loss = torch.mean((rgb_map - rgb_train) ** 2)
            total_loss = loss


            # regularization
            # TV loss on the density planes
            if self.cfg.model.TV_weight_density > 0:
                # TV_weight_density = lr_factor * self.cfg.model.TV_weight_density
                TV_weight_density = self.cfg.model.TV_weight_density
                loss_tv = model.TV_loss_density(tvreg) * TV_weight_density
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar(
                    "train/reg_tv_density",
                    loss_tv.detach().item(),
                    global_step=iteration,
                )

            # L1 loss on the density planes
            if self.cfg.model.L1_weight_density > 0:
                # L1_weight_density = lr_factor * self.cfg.model.L1_weight_density
                L1_weight_density = 1 * self.cfg.model.L1_weight_density
                loss_l1 = model.L1_loss_density() * L1_weight_density
                total_loss = total_loss + loss_l1
                summary_writer.add_scalar(
                    "train/reg_l1_density",
                    loss_l1.detach().item(),
                    global_step=iteration,
                )

            # Loss on the rendered and gt depth maps.
            if self.cfg.model.depth_loss and self.cfg.model.depth_loss_weight > 0:
                depth_loss = (depth_map.unsqueeze(-1) - depth) ** 2
                mask = depth != 0
                depth_loss = (
                    torch.mean(depth_loss[mask]) * self.cfg.model.depth_loss_weight
                )
                total_loss += depth_loss
                summary_writer.add_scalar(
                    "train/depth_loss",
                    depth_loss.detach().item(),
                    global_step=iteration,
                )

            # Dist loss from Mip360 paper.
            if self.cfg.model.dist_loss and self.cfg.model.dist_loss_weight > 0:
                svals = (weights - model.near_far[0]) / (
                    model.near_far[1] - model.near_far[0]
                )
                dist_loss = (
                    compute_dist_loss(alphas_map[..., :-1], svals)
                    * self.cfg.model.dist_weight
                )
                total_loss += dist_loss
                summary_writer.add_scalar(
                    "train/dist_loss", dist_loss.detach().item(), global_step=iteration
                )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step(); scheduler.step()

            loss = loss.detach().item()
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
            summary_writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar("train/mse", loss, global_step=iteration)

            # Print the current values of the losses.
            if iteration % self.cfg.systems.progress_refresh_rate == 0:
                pbar.set_description(
                    f"Iteration {iteration:05d}:"
                    + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                    + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                    + f" mse = {loss:.6f}"
                    + f" lr = {optimizer.param_groups[0]['lr']:.6f}"
                )
                PSNRs = []

            # # Decay the learning rate.
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = param_group["lr_org"] * lr_factor

            # Evaluation for every self.cfg.systems.vis_every steps.
            if (
                iteration % self.cfg.systems.vis_every == self.cfg.systems.vis_every - 1
                and self.cfg.data.N_vis != 0
            ):
                PSNRs_test = render(
                    test_dataset,
                    model,
                    self.cfg,
                    f"{self.logfolder}/imgs_vis/",
                    prefix=f"{iteration:06d}_",
                    white_bg=white_bg,
                    n_samples=n_samples,
                    ndc_ray=ndc_ray,
                    device=self.device,
                    compute_extra_metrics=False,
                    progress=iteration/self.cfg.optim.n_iters,
                    is_val=True
                )

                PSNRs_disentangled = render_disentangled(
                    test_dataset,
                    model,
                    self.cfg,
                    f"{self.logfolder}/imgs_vis/",
                    prefix=f"{iteration:06d}_",
                    white_bg=white_bg,
                    n_samples=n_samples,
                    ndc_ray=ndc_ray,
                    device=self.device,
                    compute_extra_metrics=False,
                    progress=iteration/self.cfg.optim.n_iters,
                    is_val=True
                )

                summary_writer.add_scalar(
                    "test/psnr", np.mean(PSNRs_test), global_step=iteration
                )

                torch.cuda.synchronize()

            # Calculate the emptiness voxel.
            if iteration in self.cfg.model.update_alpha_grid_mask_list:
                if (
                    reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
                ):  # update volume resolution
                    reso_mask = reso_cur
                    model.update_alpha_mask(tuple(reso_mask), progress=iteration/self.cfg.optim.n_iters)

                # Shrink tensor feature maps
                if iteration == self.cfg.model.update_alpha_grid_mask_list[0]:
                    # model.shrink(new_aabb)

                    # Re-initialize L1 loss weights
                    self.cfg.model.L1_weight_density = self.cfg.model.L1_weight_density_rest
                    print("[Rescaling regularization weight] continuing L1_reg_weight", self.cfg.model.L1_weight_density)

                # Filtering rays
                if iteration == self.cfg.model.update_alpha_grid_mask_list[1]:
                    allrays, allrgbs = (train_dataset.all_rays, train_dataset.all_rgbs)
                    if self.depth_data: alldepths = train_dataset.all_depths
                    else:alldepths = None

                    allrays, allrgbs, alldepths = model.filtering_rays(
                        allrays, allrgbs, alldepths, bbox_only=True)
                    
                    train_dataset.all_rays = allrays
                    train_dataset.all_rgbs = allrgbs
                    train_dataset.all_depths = alldepths

                    # initialize the data sampler
                    self.init_sampler(train_dataset)
                    

            # Upsample the volume grid.
            if iteration in self.cfg.model.upsample_list:
                if self.cfg.model.upsampling_type == "aligned":
                    reso_cur = [reso_cur[i] * 2 - 1 for i in range(len(reso_cur))]
                else:
                    N_voxel = self.N_voxel_list.pop(0)
                    reso_cur = N_to_reso(
                        N_voxel, model.aabb, self.cfg.model.nonsquare_voxel
                    )
                
                n_samples = cal_n_samples(reso_cur, self.cfg.model.step_ratio,
                                          max_value=self.cfg.model.n_samples)
                
                model.upsample_volume_grid(reso_cur)

                grad_vars = model.get_optparam_groups(self.cfg.optim, 1.0)
                optimizer = torch.optim.Adam(
                    grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
                )
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.cfg.optim.lr_decay_target_ratio**(1/self.cfg.optim.n_iters))

        end_time = time.time()
        exec_time = end_time - start_time
        
        # Wall Clock Time
        print(f"Execution time: {end_time - start_time} seconds")
        with open(f'{self.logfolder}/exec_time.txt', 'w') as file: file.write(f'[Traing]Exec time : {exec_time}\n')
        